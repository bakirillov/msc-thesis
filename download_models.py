from google_drive_downloader import GoogleDriveDownloader as gdd


print("Downloading ecoli.pkl")
gdd.download_file_from_google_drive(
    file_id='1O19ZxEu7fPeQ5jHYGH12aZmU7ms3TdbK',
    dest_path='./indices/ecoli.pkl',
    unzip=True
)
print("Downloading models")
gdd.download_file_from_google_drive(
    file_id='1qyrn52q41cGQV2Yddn1pSMjeKToFxcbn',
    dest_path='./models/test/gecrispr.cat',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='1kio_VU01m1dKjn-l8LZDSwF3-FfAySzo',
    dest_path='./models/test/peng_difference.ptch',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='1Pc08PSrAkylMCF68wcIvQ-Pm138vjxrM',
    dest_path='./models/test/revcore_full_model_2.ptch',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='10xzvjIKCjLArEEoi4YeQX5iJs7Tx4SKG',
    dest_path='./models/test/test_background.pkl',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='12m7-Vkzdwy9IylTeOb6ve4n5L4bc0jCk',
    dest_path='./models/test/test_umap.bin',
    unzip=True
)