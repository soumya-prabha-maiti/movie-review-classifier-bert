import datetime
import io
import json
import os

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


class GDriveUtils:
    LOG_EVENTS = True

    @staticmethod
    def get_gdrive_service(creds_stringified: str | None = None):
        SCOPES = ["https://www.googleapis.com/auth/drive"]
        if not creds_stringified:
            print(
                "Attempting to use google drive creds from environment variable"
            ) if GDriveUtils.LOG_EVENTS else None
            creds_stringified = os.getenv("GOOGLE_SERVICE_ACC_CREDS")
        creds_dict = json.loads(creds_stringified)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
        return build("drive", "v3", credentials=creds)

    @staticmethod
    def upload_file_to_gdrive(
        local_file_path,
        drive_parent_folder_id: str,
        drive_filename: str | None = None,
        creds_stringified: str | None = None,
    ) -> str:
        service = GDriveUtils.get_gdrive_service(creds_stringified)

        if not drive_filename:
            drive_filename = os.path.basename(local_file_path)

        file_metadata = {
            "name": drive_filename,
            "parents": [drive_parent_folder_id],
        }
        file = (
            service.files()
            .create(body=file_metadata, media_body=local_file_path)
            .execute()
        )
        print(
            "File uploaded, drive file id: ", file.get("id")
        ) if GDriveUtils.LOG_EVENTS else None
        return file.get("id")

    @staticmethod
    def upload_file_to_gdrive_sanity_check(
        drive_parent_folder_id: str,
        creds_stringified: str | None = None,
    ):
        try:
            curr_time_utc = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"gdrive_upload_test_{curr_time_utc}_UTC.txt"
            print(
                "Creating local file to upload: ", file_name
            ) if GDriveUtils.LOG_EVENTS else None
            with open(file_name, "w") as f:
                f.write(f"gdrive_upload_test_{curr_time_utc}_UTC")
            return GDriveUtils.upload_file_to_gdrive(
                file_name, drive_parent_folder_id, creds_stringified=creds_stringified
            )
        except Exception as e:
            raise e
        finally:
            if os.path.exists(file_name):
                print(
                    "Deleting local file: ", file_name
                ) if GDriveUtils.LOG_EVENTS else None
                os.remove(file_name)

    @staticmethod
    def download_file_from_gdrive(
        drive_file_id: str,
        local_file_path: str | None = None,
        creds_stringified: str | None = None,
    ):
        service = GDriveUtils.get_gdrive_service(creds_stringified)

        drive_filename = service.files().get(fileId=drive_file_id, fields="name").execute().get('name')

        if not local_file_path:
            local_file_path = f"{drive_file_id}_{drive_filename}"

        request = service.files().get_media(fileId=drive_file_id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request, chunksize= 25 * 1024 * 1024)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Downloading gdrive file {drive_filename} to local file {local_file_path}: {int(status.progress() * 100)}%.") if GDriveUtils.LOG_EVENTS else None

        if os.path.dirname(local_file_path):
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, "wb") as f:
            f.write(file.getvalue())
        print(
            "Downloaded file locally to: ", local_file_path
        ) if GDriveUtils.LOG_EVENTS else None

    @staticmethod
    def download_file_from_gdrive_sanity_check(
        drive_parent_folder_id: str,
        creds_stringified: str | None = None,
    ):
        file_id = GDriveUtils.upload_file_to_gdrive_sanity_check(
            drive_parent_folder_id, creds_stringified
        )
        GDriveUtils.download_file_from_gdrive(
            file_id, creds_stringified=creds_stringified
        )

    def stringify_json_creds(json_file: str, txt_file: str) -> str:
        with open(json_file, "r") as f:
            creds_dict = json.load(f)
        with open(txt_file, "w") as f:
            f.write(json.dumps(creds_dict))