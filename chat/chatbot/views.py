import os
import re
import atexit
import logging
from dotenv import load_dotenv

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse

from .models import ChatThread, Message, Document

try:
    from langchain_community.vectorstores.faiss import FAISS
except Exception as e:
    print("FAISS import failed:", e)
    FAISS = None

try:
    from langchain_text_splitters import CharacterTextSplitter
except Exception as e:
    print("CharacterTextSplitter import failed:", e)
    CharacterTextSplitter = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception as e:
    print("HuggingFaceEmbeddings import failed:", e)
    HuggingFaceEmbeddings = None

try:
    from langchain_community.llms import LlamaCpp
except Exception as e:
    print("LlamaCpp import failed:", e)
    LlamaCpp = None

try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
except Exception:
    UnstructuredPDFLoader = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

load_dotenv()
logger = logging.getLogger(__name__)

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M.gguf")
LLAMA_CONTEXT_WINDOW = int(os.getenv("LLAMA_CONTEXT_WINDOW", "512"))
_llm_instance = None

def cleanup_llama():
    global _llm_instance
    try:
        if _llm_instance is not None:
            close_fn = getattr(_llm_instance, "close", None)
            if callable(close_fn):
                close_fn()
    except Exception:
        pass
    _llm_instance = None

atexit.register(cleanup_llama)

def get_llama_model():
    global _llm_instance
    if _llm_instance is None:
        if LlamaCpp is None:
            logger.warning("LlamaCpp wrapper not available.")
            return None
        try:
            _llm_instance = LlamaCpp(model_path=LLAMA_MODEL_PATH, n_ctx=LLAMA_CONTEXT_WINDOW)
        except Exception as e:
            logger.exception("Failed to initialize LlamaCpp: %s", e)
            _llm_instance = None
    return _llm_instance

def clean_assistant_text(text):
    if not text:
        return ""
    s = str(text)
    patterns = [r"<<\s*/?\s*INST\s*>>", r"\[\s*/?\s*INST\s*\]", r"<<\s*/?\s*SYS\s*>>", r"<\s*/\s*s\s*>"]
    for p in patterns:
        s = re.sub(p, " ", s, flags=re.IGNORECASE)
    s = s.replace("You are a helpful assistant.", " ")
    s = " ".join(s.split()).strip()
    return s

def extract_text_from_pdf_with_pypdf(doc_path):
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Install with: pip install pypdf")
    pages = []
    reader = PdfReader(doc_path)
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join([p for p in pages if p])

def extract_text_from_pdf_unstructured(doc_path):
    if UnstructuredPDFLoader is None:
        raise RuntimeError("UnstructuredPDFLoader not available.")
    loader = UnstructuredPDFLoader(doc_path)
    pages = loader.load()
    return "\n".join([p.page_content for p in pages])

def load_text_from_file(doc_path, filetype='txt'):
    if filetype == 'pdf':
        try:
            return extract_text_from_pdf_with_pypdf(doc_path)
        except Exception as e_pypdf:
            logger.warning("pypdf extraction failed (%s); trying Unstructured fallback...", e_pypdf)
            try:
                return extract_text_from_pdf_unstructured(doc_path)
            except Exception as e_unstruct:
                raise RuntimeError(
                    "Failed to extract text from PDF. Install pypdf (pip install pypdf) "
                    "or make sure Unstructured + Poppler are installed. "
                    f"Details: pypdf error: {e_pypdf}; unstructured error: {e_unstruct}"
                )
    else:
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def prepare_doc_retriever(doc_path, filetype='txt', chunk_size=500, chunk_overlap=50):
    if CharacterTextSplitter is None or FAISS is None or HuggingFaceEmbeddings is None:
        raise RuntimeError("Document QA dependencies missing. Install langchain_community, faiss-cpu, langchain-text-splitters, sentence-transformers.")
    text = load_text_from_file(doc_path, filetype=filetype)
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_texts(chunks, embeddings)
    retriever = vector_db.as_retriever()
    return retriever

def estimate_tokens_simple(text: str) -> int:
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)

def get_llama_response(user_message, chat_history=None, model_context_window=LLAMA_CONTEXT_WINDOW, requested_max_response=150):
    llm = get_llama_model()
    if llm is None:
        return "Model not available. Check server logs."
    system_prompt = "You are a helpful assistant. Reply concisely and directly to the user's question only."
    prompt = system_prompt + "\n"
    if chat_history and len(chat_history) > 0:
        last = chat_history[-1]
        prompt += f"User: {last['user']}\n"
        prompt += f"Assistant: {last['bot']}\n"
    prompt += f"User: {user_message}\nAssistant:"
    prompt_tokens = estimate_tokens_simple(prompt)
    available_for_response = model_context_window - prompt_tokens - 8
    safe_max_tokens = max(16, min(requested_max_response, available_for_response))
    if available_for_response < 16:
        prompt = system_prompt + f"\nUser: {user_message}\nAssistant:"
        prompt_tokens = estimate_tokens_simple(prompt)
        available_for_response = model_context_window - prompt_tokens - 8
        safe_max_tokens = max(16, min(requested_max_response, available_for_response))
    try:
        answer = llm.invoke(prompt) if llm else "Model not available."
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return f"Model error during generation. Try again. Details: {str(e)}"
    return clean_assistant_text(str(answer)).strip()

def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            if User.objects.filter(username=username).exists():
                messages.info(request, "Account already exists. Please sign in.")
                return redirect("login")
            form.save()
            messages.success(request, "Account created successfully. Please log in.")
            return redirect("login")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect("chat")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})

def logout_view(request):
    logout(request)
    return redirect("login")

def home_view(request):
    chat_history = request.session.get('chat_history', [])
    bot_response = None
    if request.method == "POST":
        if "clear_chat" in request.POST:
            chat_history = []
            request.session['chat_history'] = []
            bot_response = "Chat history cleared."
        else:
            user_message = request.POST.get("user_input", "").strip()
            if user_message:
                bot_response = get_llama_response(user_message, chat_history)
                chat_history.append({'user': user_message, 'bot': clean_assistant_text(bot_response)})
                request.session['chat_history'] = chat_history
            else:
                bot_response = "Please type something!"
        return render(request, 'home.html', {
            'chat_history': chat_history,
            'bot_response': bot_response,
        })
    return render(request, 'home.html', {
        'chat_history': chat_history,
        'bot_response': bot_response,
    })

@login_required
def chat_view(request):
    threads = ChatThread.objects.filter(user=request.user)
    selected_thread_id = request.GET.get("thread")
    selected_thread = ChatThread.objects.filter(id=selected_thread_id, user=request.user).first() if selected_thread_id else None
    thread_messages_qs = selected_thread.messages.all() if selected_thread else []
    if request.method == "POST" and "delete_thread" in request.POST and selected_thread:
        Message.objects.filter(thread=selected_thread).delete()
        selected_thread.delete()
        return redirect("chat")
    if request.method == "POST" and selected_thread and request.POST.get("message"):
        user_msg = request.POST.get("message").strip()
        chat_history = [
            {"user": m.content, "bot": "" if m.sender == "user" else clean_assistant_text(m.content)}
            for m in selected_thread.messages.all()
        ]
        bot_msg = get_llama_response(user_msg, chat_history)
        Message.objects.create(thread=selected_thread, sender="user", content=user_msg)
        Message.objects.create(thread=selected_thread, sender="bot", content=bot_msg)
        return redirect(f"/chat?thread={selected_thread.id}")
    if request.method == "POST" and "new_thread" in request.POST:
        new_thread = ChatThread.objects.create(user=request.user, title=f"Chat {threads.count() + 1}")
        return redirect(f"/chat?thread={new_thread.id}")
    return render(request, "chat.html", {
        "threads": threads,
        "thread_messages": thread_messages_qs,
        "selected_thread": selected_thread
    })

@login_required
def upload_document(request):
    message = ""
    if request.method == "POST" and request.FILES.get("document"):
        doc = Document(
            user=request.user,
            file=request.FILES["document"],
            title=request.FILES["document"].name
        )
        doc.save()
        messages.success(request, "File uploaded successfully.")
        return redirect("documents")
    docs = Document.objects.filter(user=request.user)
    return render(request, "documents.html", {
        "documents": docs,
        "message": message,
    })

@login_required
def delete_document(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    if request.method == "POST":
        if doc.file:
            doc.file.delete(save=False)
        doc.delete()
        messages.success(request, "Document deleted.")
        return redirect("documents")
    return render(request, "confirm_delete_document.html", {"document": doc})

@login_required
def download_document(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    response = HttpResponse(doc.file, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{doc.title}"'
    return response

@login_required
def ask_doc_question(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)
    doc_id = request.POST.get("doc_id")
    question = request.POST.get("question", "").strip()
    if not doc_id:
        return JsonResponse({"error": "doc_id not provided"}, status=400)
    if not question:
        return JsonResponse({"error": "question not provided"}, status=400)
    try:
        document = Document.objects.get(id=doc_id, user=request.user)
    except Document.DoesNotExist:
        return JsonResponse({"error": "Document not found."}, status=404)
    try:
        doc_path = document.file.path
        ext = os.path.splitext(doc_path)[-1].lower()
        filetype = 'pdf' if ext == '.pdf' else 'txt'
        text = load_text_from_file(doc_path, filetype=filetype)
        if "first line" in question.lower():
            first_line = next((line for line in text.splitlines() if line.strip()), "No line found.")
            answer = f"The first line is: {first_line}"
        else:
            snippet = "\n".join(text.splitlines()[:10]) 
            prompt = f"Document snippet:\n{snippet}\n\nQuestion: {question}\nAnswer concisely:"
            llm = get_llama_model()
            answer = llm.invoke(prompt) if llm else "Model not available."
        if not answer or answer.strip() == "":
            answer = "Sorry, no answer could be generated from the document."
    except Exception as e:
        logger.exception("ask_doc_question error: %s", e)
        answer = f"Bot error: {str(e)}"
    return JsonResponse({"answer": answer})

@login_required
def delete_all_chats(request):
    if request.method == "POST":
        ChatThread.objects.filter(user=request.user).delete()
        Message.objects.filter(thread__user=request.user).delete()
        return JsonResponse({"status": "success", "message": "All chats deleted."})
    return JsonResponse({"error": "Invalid request"}, status=400)
