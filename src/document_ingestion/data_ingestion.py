from __future__ import annotations
import base64
import io
import os
import sys
import json
# from turtle import pd
import uuid
import hashlib
from instructor import Image
import pandas as pd
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files, encode_image_to_base64,preprocess_image
from utils.document_ops import load_documents
import sqlite3
import textwrap

from langchain_core.messages import HumanMessage
import imageio


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".ppt", ".pptx", ".xlsx", ".csv", ".sql", ".jpg", ".png", ".jpeg", ".gif", ".tiff", ".bmp", ".webp", ".svg"}
IMAGE_FILES = {".jpg", ".png", ".jpeg", ".gif", ".tiff", ".bmp", ".webp", ".svg"}


# FAISS Manager (load-or-create)
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}} ## this is dict of rows
        
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists
        

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        
    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        
    def add_documents(self,docs: List[Document]):
        
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        
        new_docs: List[Document] = []
        
        for d in docs:
            
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
            
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
        
        
class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()
           
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
           
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager very important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

    

class DocHandler:
    """
    File save + read (page-wise) for analysis.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        self.model_loader = ModelLoader()
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_file(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            # if not (filename.endswith(tuple(SUPPORTED_EXTENSIONS)) or filename.endswith(tuple(IMAGE_FILES))):
                # raise ValueError("Invalid file type. Only supported file types are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("File saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save file", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save file: {str(e)}", e) from e

    def read_file(self, file_path: str) -> str:
        try:
       
            if Path(file_path).suffix.lower() == ".csv":
                 return self.read_csv_file(file_path)
            if Path(file_path).suffix.lower() == ".sql":
                 return self.execute_sql_file(file_path)
            if Path(file_path).suffix.lower() in  tuple(IMAGE_FILES):
                 return self.read_image_file(file_path)
            text_chunks = []
            with fitz.open(file_path) as doc:
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")  # type: ignore
            text = "\n".join(text_chunks)
            log.info("File read successfully", file_path=file_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read file", error=str(e), file_path=file_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process file: {file_path}", e) from e
    
    def read_csv_file(self, file_path: str) -> str:
        try:
            
            text_chunks = []
            df = pd.read_csv(file_path)
            text_chunks = []

            for i in range(0, len(df), 10):  # step by 10
                page = df.iloc[i:i+10]
                text_chunks.append(f"\n--- Page {i//10 + 1} ---\n{page.to_string(index=False)}")
            text = "\n".join(text_chunks)
            log.info("File read successfully", file_path=file_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read file", error=str(e), file_path=file_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process file: {file_path}", e) from e
        
    def execute_sql_file(self,file_path: str):
        connection = sqlite3.connect('my_database.db')
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        cursor = connection.cursor()
        cursor.executescript(sql_script)  # For SQLite
        connection.commit()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT name
            FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = cursor.fetchall()  # Get all table names
    
        documents: list[Document] = []

        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                for idx, row in enumerate(rows[:10], start=1):
                    # Format each column-value pair with indentation for clarity
                    content_lines = [f"{col}: {val}" for col, val in zip(columns, row)]
                    content = "\n".join(content_lines)

                    # Create Document with metadata including row index
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source_table": table_name,
                            "row_index": idx
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Failed to fetch data from table '{table_name}': {e}")
        cursor.close()    
        connection.close()
        return documents

   
  

    def read_image_file(self, file_path: str) -> str:
        try:
            # Load image
            pil_image = preprocess_image(file_path) #Image.open(file_path)
            image_base64 = encode_image_to_base64(image=pil_image, format=pil_image.format or "PNG")

            # Optional: Embed as data URI
            # data_uri = f"data:image/png;base64,{image_base64}"

            # Construct multimodal message
            mesage = HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Extract all visible text from this image. Include handwritten and printed text."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ])
            model = self.model_loader.load_img_reader_llm()
            response = model.invoke([mesage])
            raw_text = response.content

            # # Extract text using OCR
            # model = self.model_loader.load_img_reader_llm()
            # raw_text = model.generate_content([
            #             image,
            #             "Extract all visible text from this image. Include handwritten and printed text."
            #         ])

            # raw_text = pytesseract.image_to_string(image)

            # Split text into chunks of ~500 characters (adjust as needed)
            wrapped_chunks = textwrap.wrap(raw_text, width=500)
            text_chunks = []

            for i, chunk in enumerate(wrapped_chunks):
                text_chunks.append(f"\n--- Page {i + 1} ---\n{chunk}")

            text = "\n".join(text_chunks)

            log.info("Image read successfully", file_path=file_path, session_id=self.session_id, pages=len(text_chunks))
            return text

        except Exception as e:
            log.error("Failed to read image", error=str(e), file_path=file_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process image: {file_path}", e) from e

class DocumentComparator:
    """
    Save, read & combine PDFs for comparison with session-based versioning.
    """
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.model_loader = ModelLoader()
        log.info("DocumentComparator initialized", session_path=str(self.session_path))
       
    
    def save_file(self, uploaded_file) -> str:
        try:
            filename = os.path.basename(uploaded_file.name)
            # if not (filename.endswith(tuple(SUPPORTED_EXTENSIONS)) or filename.endswith(tuple(IMAGE_FILES))):
                # raise ValueError("Invalid file type. Only supported file types are allowed.")
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
            log.info("File saved successfully", file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save file", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save file: {str(e)}", e) from e  
         
    def save_uploaded_files(self, reference_file, actual_file):
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name

            self.save_file(reference_file)
            self.save_file(actual_file)
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
            # for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
            #     filename = getattr(fobj, "name", None)
            #     if filename and Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            #         raise ValueError("Only supported file types are allowed.")
                 
            #     with open(out, "wb") as f:
            #         if hasattr(fobj, "read"):
            #             f.write(fobj.read())
            #         else:
            #             f.write(fobj.getbuffer())  # fallback


            # log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            # return ref_path, act_path
        except Exception as e:
            log.error("Error saving  files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e

    def read_file(self, file_path: Path) -> str:
        try:
            if Path(file_path).suffix.lower() == ".csv":
                return self.read_csv_file(file_path)
            if Path(file_path).suffix.lower() == ".sql":
                 return self.execute_sql_file(file_path)
            if Path(file_path).suffix.lower() in  tuple(IMAGE_FILES) :
                return self.read_image_file(file_path)
            with fitz.open(file_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"File is encrypted: {file_path.name}")
                parts = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()  # type: ignore
                    if text.strip():
                        parts.append(f"\n --- Page {page_num + 1} --- \n{text}")
            log.info("File read successfully", file=str(file_path), pages=len(parts))
            return "\n".join(parts)
        except Exception as e:
            log.error("Error reading file", file=str(file_path), error=str(e))
            raise DocumentPortalException("Error reading file", e) from e

    def read_csv_file(self, file_path: str) -> str:
        try:
            
            text_chunks = []
            df = pd.read_csv(file_path)
            text_chunks = []

            for i in range(0, len(df), 10):  # step by 10
                page = df.iloc[i:i+10]
                text_chunks.append(f"\n--- Page {i//10 + 1} ---\n{page.to_string(index=False)}")
            text = "\n".join(text_chunks)
            log.info("File read successfully", file_path=file_path, session_id=self.session_id, pages=len(text_chunks))
            return text
        except Exception as e:
            log.error("Failed to read file", error=str(e), file_path=file_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process file: {file_path}", e) from e
        
    def execute_sql_file(self,file_path: str):
        connection = sqlite3.connect('my_database.db')
        with open(file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        cursor = connection.cursor()
        cursor.executescript(sql_script)  # For SQLite
        connection.commit()
        cursor = connection.cursor()
        cursor.execute("""
            SELECT name
            FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        tables = cursor.fetchall()  # Get all table names
    
        documents: list[Document] = []

        for table in tables:
            table_name = table[0]
            try:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                for idx, row in enumerate(rows[:10], start=1):
                    # Format each column-value pair with indentation for clarity
                    content_lines = [f"{col}: {val}" for col, val in zip(columns, row)]
                    content = "\n".join(content_lines)

                    # Create Document with metadata including row index
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source_table": table_name,
                            "row_index": idx
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                print(f"Failed to fetch data from table '{table_name}': {e}")
        cursor.close()    
        connection.close()
        return documents   
    
    def read_image_file(self, file_path: str) -> str:
        try:
            # Load image
            pil_image = preprocess_image(file_path)
            image_base64 = encode_image_to_base64(image=pil_image, format=pil_image.format or "PNG")

            # Optional: Embed as data URI
            # data_uri = f"data:image/png;base64,{image_base64}"

            # Construct multimodal message
            mesage = HumanMessage(content=[
                {
                    "type": "text",
                    "text": "Extract all visible text from this image. Include handwritten and printed text."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ])
            model = self.model_loader.load_img_reader_llm()
            response = model.invoke([mesage])
            raw_text = response.content

            # # Extract text using OCR
            # model = self.model_loader.load_img_reader_llm()
            # raw_text = model.generate_content([
            #             image,
            #             "Extract all visible text from this image. Include handwritten and printed text."
            #         ])

            # raw_text = pytesseract.image_to_string(image)

            # Split text into chunks of ~500 characters (adjust as needed)
            wrapped_chunks = textwrap.wrap(raw_text, width=500)
            text_chunks = []

            for i, chunk in enumerate(wrapped_chunks):
                text_chunks.append(f"\n--- Page {i + 1} ---\n{chunk}")

            text = "\n".join(text_chunks)

            log.info("Image read successfully", file_path=file_path, session_id=self.session_id, pages=len(text_chunks))
            return text

        except Exception as e:
            log.error("Failed to read image", error=str(e), file_path=file_path, session_id=self.session_id)
            raise DocumentPortalException(f"Could not process image: {file_path}", e) from e
        
    def combine_documents(self) -> str:
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:               
                    content = self.read_file(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int = 3):
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e

# if __name__ == "__main__":
    #Path("sample.docx"), Path("sample.xlsx"), Path("sample.pptx"), Path("image1.png"
    # ing = ChatIngestor()
    # ing.built_retriver(paths)
    # paths = [Path("D:\\LLMOPS Industry Projects\\document_portal\\data\\image_text.webp"), Path("D:\\LLMOPS Industry Projects\\document_portal\\data\\image_text.png")]
    # doc= DocHandler ()
    # text= doc.read_file(paths[0])
    # print(text)
    # paths = [Path("D:\\LLMOPS Industry Projects\\document_portal\\data\\english paper pattern.pdf"), Path("D:\\LLMOPS Industry Projects\\document_portal\\data\\images with text\\image_text2.png")]
    # doc = DocumentComparator()
    # ref_path, act_path = doc.save_uploaded_files(paths[0],paths[1])
    # combined_text = doc.combine_documents()
    # comp = DocumentComparatorLLM()
    # text = comp.compare_documents(combined_text)
    # print(text)
