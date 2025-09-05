from __future__ import annotations
import os
import re
from unicodedata import name
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
from typing import Iterable, List
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".ppt", ".pptx", ".xlsx", ".csv", ".sql", ".jpg", ".png", ".jpeg", ".gif", ".tiff", ".bmp", ".webp", ".svg"}
IMAGE_FILES = {".jpg", ".png", ".jpeg", ".gif", ".tiff", ".bmp", ".webp", ".svg"}

# ----------------------------- #
# Helpers (file I/O + loading)  #
# ----------------------------- #
def generate_session_id(prefix: str = "session") -> str:
    ist = ZoneInfo("Asia/Kolkata")
    return f"{prefix}_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

def save_file(uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            # if not (filename.endswith(tuple(SUPPORTED_EXTENSIONS)) or filename.endswith(tuple(IMAGE_FILES))):
                # raise ValueError("Invalid file type. Only supported file types are allowed.")
            save_path = os.path.join(filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("File saved successfully", file=filename, save_path=save_path)
            return save_path
        except Exception as e:
            log.error("Failed to save file", error=str(e))
            raise DocumentPortalException(f"Failed to save file: {str(e)}", e) from e  
         
def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            ref_path = save_file(uf)
            saved.append(Path(ref_path))
            log.info("Files saved for ingestion", uploaded=uf, saved_as=str(ref_path))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e
    
def save_uploaded_files_old(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """Save uploaded files (Streamlit-like) and return local paths."""
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for uf in uploaded_files:
            name = getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue
            # Clean file name (only alphanum, dash, underscore)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"
            fname = f"{uuid.uuid4().hex[:8]}{ext}"
            out = target_dir / fname
            with open(out, "wb") as f:
                if hasattr(uf, "read"):
                    f.write(uf.read())
                else:
                    f.write(uf.getbuffer())  # fallback
            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))
        return saved
    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e