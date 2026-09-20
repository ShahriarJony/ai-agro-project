"""
AgroAI - Firebase Firestore Database Service
Connects directly to Google Cloud Firestore using Firebase Admin SDK when credentials are present,
or via Firestore REST API with Firebase Auth session when operating without service account key.
Serves real records from live Firestore for Farms, Fields, and Audit Logs.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

logger = logging.getLogger("AgroAI.Firebase")

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Potential key paths for Firebase Service Account Key JSON
POSSIBLE_KEY_PATHS = [
    os.environ.get("FIREBASE_SERVICE_ACCOUNT"),
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    os.path.join(BACKEND_DIR, "serviceAccountKey.json"),
    os.path.join(BACKEND_DIR, "firebase-credentials.json"),
    os.path.join(PROJECT_ROOT, "serviceAccountKey.json"),
    os.path.join(PROJECT_ROOT, "firebase-credentials.json"),
]

# Firebase Project Configuration
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID") or "agroai-b72ec"
FIREBASE_API_KEY = (
    os.environ.get("FIREBASE_API_KEY")
    or os.environ.get("VITE_FIREBASE_API_KEY")
    or "AIzaSyC3wiPdZQ3NTocZp6cqjzQb14EIHwzAE9E"
)

firestore_client = None
using_admin_sdk = False
using_firestore = False

# Helper for REST authentication
_cached_auth_token = None
_cached_uid = None


def _init_firestore_admin():
    global firestore_client, using_admin_sdk, using_firestore
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if firebase_admin._apps:
            try:
                firestore_client = firestore.client()
                using_admin_sdk = True
                using_firestore = True
                print("[AgroAI Firebase] Connected to existing Firebase Admin SDK instance.")
                return True
            except Exception as e:
                logger.warning(f"Existing Firebase app client error: {e}")

        key_file = None
        for p in POSSIBLE_KEY_PATHS:
            if p and os.path.exists(p):
                key_file = p
                break

        if key_file:
            try:
                cred = credentials.Certificate(key_file)
                firebase_admin.initialize_app(cred)
                firestore_client = firestore.client()
                using_admin_sdk = True
                using_firestore = True
                print(f"[AgroAI Firebase] Connected to live Firestore DB using key file: {key_file}")
                return True
            except Exception as e:
                logger.warning(f"Key file initialization failed ({key_file}): {e}")

        # Environment JSON string
        env_json = (
            os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
            or (os.environ.get("FIREBASE_SERVICE_ACCOUNT") if os.environ.get("FIREBASE_SERVICE_ACCOUNT", "").startswith("{") else None)
        )
        if env_json:
            try:
                cert_dict = json.loads(env_json)
                cred = credentials.Certificate(cert_dict)
                firebase_admin.initialize_app(cred)
                firestore_client = firestore.client()
                using_admin_sdk = True
                using_firestore = True
                print("[AgroAI Firebase] Connected to live Firestore DB using environment JSON credentials.")
                return True
            except Exception as e:
                logger.warning(f"Environment JSON initialization failed: {e}")

        # Individual env vars: FIREBASE_CLIENT_EMAIL & FIREBASE_PRIVATE_KEY
        client_email = os.environ.get("FIREBASE_CLIENT_EMAIL")
        private_key = os.environ.get("FIREBASE_PRIVATE_KEY")
        if client_email and private_key:
            try:
                pk_clean = private_key.replace("\\n", "\n")
                cert_dict = {
                    "type": "service_account",
                    "project_id": FIREBASE_PROJECT_ID,
                    "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID", "default_key_id"),
                    "private_key": pk_clean,
                    "client_email": client_email,
                    "client_id": os.environ.get("FIREBASE_CLIENT_ID", "default_client_id"),
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
                cred = credentials.Certificate(cert_dict)
                firebase_admin.initialize_app(cred)
                firestore_client = firestore.client()
                using_admin_sdk = True
                using_firestore = True
                print("[AgroAI Firebase] Connected to live Firestore DB using client_email & private_key env vars.")
                return True
            except Exception as e:
                logger.warning(f"Individual env vars initialization failed: {e}")

    except Exception as e:
        logger.info(f"Firebase Admin SDK notice: {e}")

    using_admin_sdk = False
    using_firestore = True  # Fallback to REST Client
    return False


# Attempt Admin SDK initialization first
_init_firestore_admin()


# REST API Fallback Helpers
def _get_rest_auth():
    global _cached_auth_token, _cached_uid
    if _cached_auth_token and _cached_uid:
        return _cached_auth_token, _cached_uid

    url_signup = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_API_KEY}"
    payload = json.dumps(
        {
            "email": "system_backend_service@agroai.edu",
            "password": "BackendPassword123!",
            "returnSecureToken": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url_signup, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        res = urllib.request.urlopen(req)
        d = json.loads(res.read().decode("utf-8"))
        _cached_auth_token = d["idToken"]
        _cached_uid = d["localId"]
        return _cached_auth_token, _cached_uid
    except Exception:
        url_login = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        req_login = urllib.request.Request(
            url_login, data=payload, headers={"Content-Type": "application/json"}
        )
        res_login = urllib.request.urlopen(req_login)
        d = json.loads(res_login.read().decode("utf-8"))
        _cached_auth_token = d["idToken"]
        _cached_uid = d["localId"]
        return _cached_auth_token, _cached_uid


def _parse_rest_value(val: dict):
    if "stringValue" in val:
        return val["stringValue"]
    if "integerValue" in val:
        v = val["integerValue"]
        return (
            int(v)
            if str(v).isdigit() or (str(v).startswith("-") and str(v)[1:].isdigit())
            else float(v)
        )
    if "doubleValue" in val:
        return float(val["doubleValue"])
    if "booleanValue" in val:
        return bool(val["booleanValue"])
    if "mapValue" in val:
        return {k: _parse_rest_value(v) for k, v in val["mapValue"].get("fields", {}).items()}
    if "arrayValue" in val:
        return [_parse_rest_value(v) for v in val["arrayValue"].get("values", [])]
    return None


def _encode_rest_value(val: Any) -> dict:
    if val is None:
        return {"nullValue": None}
    if isinstance(val, bool):
        return {"booleanValue": val}
    if isinstance(val, int):
        return {"integerValue": str(val)}
    if isinstance(val, float):
        return {"doubleValue": val}
    if isinstance(val, str):
        return {"stringValue": val}
    if isinstance(val, dict):
        return {"mapValue": {"fields": {k: _encode_rest_value(v) for k, v in val.items()}}}
    if isinstance(val, list):
        return {"arrayValue": {"values": [_encode_rest_value(v) for v in val]}}
    return {"stringValue": str(val)}


def _rest_get_collection(collection_name: str) -> List[Dict[str, Any]]:
    token, _ = _get_rest_auth()
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{collection_name}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        res = urllib.request.urlopen(req)
        raw = json.loads(res.read().decode("utf-8"))
        items = []
        for doc in raw.get("documents", []):
            doc_name = doc.get("name", "")
            doc_id = doc_name.split("/")[-1] if doc_name else ""
            parsed = {k: _parse_rest_value(v) for k, v in doc.get("fields", {}).items()}
            f_id = parsed.get("fieldId") or parsed.get("farmId") or parsed.get("id") or doc_id
            parsed["id"] = f_id
            if collection_name == "fields":
                parsed["fieldId"] = f_id
            elif collection_name == "farms":
                parsed["farmId"] = f_id
            items.append(parsed)
        return items
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise e


def _rest_save_document(collection_name: str, doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    token, uid = _get_rest_auth()
    if "ownerId" not in data and "userId" not in data:
        data["ownerId"] = uid
        data["userId"] = uid
    payload = {"fields": {k: _encode_rest_value(v) for k, v in data.items()}}
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{collection_name}?documentId={doc_id}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req)
        return data
    except urllib.error.HTTPError as e:
        if e.code == 409 or e.code == 400:  # Already exists, patch instead
            url_patch = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{collection_name}/{doc_id}"
            req_patch = urllib.request.Request(
                url_patch,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="PATCH",
            )
            urllib.request.urlopen(req_patch)
            return data
        raise e


def _rest_update_document(collection_name: str, doc_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Partial update: only patches the keys provided in `updates`, preserving all other fields.
    Uses Firestore REST updateMask.fieldPaths to avoid full document replacement.
    """
    token, _ = _get_rest_auth()
    # Build payload containing ONLY the fields being updated
    payload = {"fields": {k: _encode_rest_value(v) for k, v in updates.items()}}
    # updateMask ensures Firestore patches only these keys; existing keys are untouched
    field_mask = "&".join(f"updateMask.fieldPaths={k}" for k in updates.keys())
    url = (
        f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}"
        f"/databases/(default)/documents/{collection_name}/{doc_id}?{field_mask}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="PATCH",
    )
    try:
        res = urllib.request.urlopen(req)
        raw = json.loads(res.read().decode("utf-8"))
        parsed = {k: _parse_rest_value(v) for k, v in raw.get("fields", {}).items()}
        parsed["id"] = doc_id
        return parsed
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(
                status_code=404, detail=f"{collection_name[:-1].capitalize()} not found"
            )
        raise e


def _rest_delete_document(collection_name: str, doc_id: str) -> Dict[str, str]:
    token, _ = _get_rest_auth()
    url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{collection_name}/{doc_id}"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}, method="DELETE"
    )
    try:
        urllib.request.urlopen(req)
        return {"message": f"{collection_name[:-1].capitalize()} {doc_id} deleted successfully."}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise HTTPException(
                status_code=404, detail=f"{collection_name[:-1].capitalize()} not found"
            )
        raise e


class AgroDatabaseService:
    @staticmethod
    def get_farms() -> List[Dict[str, Any]]:
        if using_admin_sdk and firestore_client:
            try:
                docs = firestore_client.collection("farms").stream()
                farms = []
                for d in docs:
                    item = d.to_dict()
                    f_id = item.get("farmId") or item.get("id") or d.id
                    item["id"] = f_id
                    item["farmId"] = f_id
                    farms.append(item)
                return farms
            except Exception as e:
                logger.error(f"Firestore Admin get_farms error: {e}")

        # REST API Fallback
        try:
            return _rest_get_collection("farms")
        except Exception as e:
            logger.error(f"Firestore REST get_farms error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to read data from Firebase Firestore."
            )

    @staticmethod
    def save_farm(farm_data: Dict[str, Any]) -> Dict[str, Any]:
        farm_id = (
            farm_data.get("id")
            or farm_data.get("farmId")
            or f"farm-{int(os.urandom(4).hex(), 16)}"
        )
        farm_data["id"] = farm_id
        farm_data["farmId"] = farm_id

        if using_admin_sdk and firestore_client:
            try:
                firestore_client.collection("farms").document(farm_id).set(farm_data, merge=True)
                return farm_data
            except Exception as e:
                logger.error(f"Firestore Admin save_farm error: {e}")

        try:
            return _rest_save_document("farms", farm_id, farm_data)
        except Exception as e:
            logger.error(f"Firestore REST save_farm error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to write data to Firebase Firestore."
            )

    @staticmethod
    def update_farm(farm_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates["id"] = farm_id
        updates["farmId"] = farm_id

        if using_admin_sdk and firestore_client:
            try:
                doc_ref = firestore_client.collection("farms").document(farm_id)
                doc_snap = doc_ref.get()
                if not doc_snap.exists:
                    raise HTTPException(status_code=404, detail="Farm not found")

                doc_ref.set(updates, merge=True)
                existing = doc_snap.to_dict() or {}
                existing.update(updates)
                return existing
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Firestore Admin update_farm error: {e}")

        try:
            return _rest_update_document("farms", farm_id, updates)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Firestore REST update_farm error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to update data in Firebase Firestore."
            )

    @staticmethod
    def delete_farm(farm_id: str) -> Dict[str, str]:
        if using_admin_sdk and firestore_client:
            try:
                doc_ref = firestore_client.collection("farms").document(farm_id)
                doc_snap = doc_ref.get()
                if not doc_snap.exists:
                    raise HTTPException(status_code=404, detail="Farm not found")

                doc_ref.delete()
                return {"message": f"Farm {farm_id} deleted successfully."}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Firestore Admin delete_farm error: {e}")

        try:
            return _rest_delete_document("farms", farm_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Firestore REST delete_farm error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to delete data from Firebase Firestore."
            )

    @staticmethod
    def get_fields() -> List[Dict[str, Any]]:
        if using_admin_sdk and firestore_client:
            try:
                docs = firestore_client.collection("fields").stream()
                fields = []
                for d in docs:
                    item = d.to_dict()
                    f_id = item.get("fieldId") or item.get("id") or d.id
                    item["id"] = f_id
                    item["fieldId"] = f_id
                    fields.append(item)
                return fields
            except Exception as e:
                logger.error(f"Firestore Admin get_fields error: {e}")

        try:
            return _rest_get_collection("fields")
        except Exception as e:
            logger.error(f"Firestore REST get_fields error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to read data from Firebase Firestore."
            )

    @staticmethod
    def save_field(field_data: Dict[str, Any]) -> Dict[str, Any]:
        field_id = (
            field_data.get("id")
            or field_data.get("fieldId")
            or f"field_{int(os.urandom(4).hex(), 16)}"
        )
        field_data["id"] = field_id
        field_data["fieldId"] = field_id

        if using_admin_sdk and firestore_client:
            try:
                firestore_client.collection("fields").document(field_id).set(field_data, merge=True)
                return field_data
            except Exception as e:
                logger.error(f"Firestore Admin save_field error: {e}")

        try:
            return _rest_save_document("fields", field_id, field_data)
        except Exception as e:
            logger.error(f"Firestore REST save_field error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to write data to Firebase Firestore."
            )

    @staticmethod
    def update_field(field_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        updates["id"] = field_id
        updates["fieldId"] = field_id

        if using_admin_sdk and firestore_client:
            try:
                doc_ref = firestore_client.collection("fields").document(field_id)
                doc_snap = doc_ref.get()
                if not doc_snap.exists:
                    raise HTTPException(status_code=404, detail="Field not found")

                doc_ref.set(updates, merge=True)
                existing = doc_snap.to_dict() or {}
                existing.update(updates)
                return existing
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Firestore Admin update_field error: {e}")

        try:
            return _rest_update_document("fields", field_id, updates)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Firestore REST update_field error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to update data in Firebase Firestore."
            )

    @staticmethod
    def delete_field(field_id: str) -> Dict[str, str]:
        if using_admin_sdk and firestore_client:
            try:
                doc_ref = firestore_client.collection("fields").document(field_id)
                doc_snap = doc_ref.get()
                if not doc_snap.exists:
                    raise HTTPException(status_code=404, detail="Field not found")

                doc_ref.delete()
                return {"message": f"Field {field_id} deleted successfully."}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Firestore Admin delete_field error: {e}")

        try:
            return _rest_delete_document("fields", field_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Firestore REST delete_field error: {e}")
            raise HTTPException(
                status_code=500, detail="Unable to delete data from Firebase Firestore."
            )

    @staticmethod
    def get_logs() -> List[Dict[str, Any]]:
        if using_admin_sdk and firestore_client:
            try:
                docs = firestore_client.collection("activity_logs").stream()
                logs = []
                for d in docs:
                    item = d.to_dict()
                    item["id"] = d.id
                    logs.append(item)
                return logs
            except Exception as e:
                logger.error(f"Firestore Admin get_logs error: {e}")

        try:
            return _rest_get_collection("activity_logs")
        except Exception:
            return []

    @staticmethod
    def save_log(log_data: Dict[str, Any]) -> Dict[str, Any]:
        log_id = log_data.get("id") or f"rec-{int(os.urandom(4).hex(), 16)}"
        log_data["id"] = log_id

        if using_admin_sdk and firestore_client:
            try:
                firestore_client.collection("activity_logs").document(log_id).set(log_data, merge=True)
                return log_data
            except Exception as e:
                logger.error(f"Firestore Admin save_log error: {e}")

        try:
            return _rest_save_document("activity_logs", log_id, log_data)
        except Exception:
            return log_data


