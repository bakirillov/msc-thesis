from google_drive_downloader import GoogleDriveDownloader as gdd


print("Downloading ecoli.pkl")
gdd.download_file_from_google_drive(
    file_id='1O19ZxEu7fPeQ5jHYGH12aZmU7ms3TdbK',
    dest_path='./indices/ecoli.pkl',
    unzip=True
)
print("Downloading models")
gdd.download_file_from_google_drive(
    file_id='1EpbM_G_i_p582H2FMAAuE5x_XaUgSy3N',
    dest_path='./models/test/gecrispr.cat',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='1K9_D0oQjesGFY2Y6R_BPqhzpxzBRuZCA',
    dest_path='./models/test/peng_siamese.ptch',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='1hk0y6D9NrnSwCn9DXfCel8BxKkCJAys8',
    dest_path='./models/test/revcore_full_model_2.ptch',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='1sr538gO_5_QQSax1zrwcbzh6ZrLrYzui',
    dest_path='./models/test/test_background.pkl',
    unzip=True
)
gdd.download_file_from_google_drive(
    file_id='1nRF0pVnoV0hMehQlBAIf9QSErfdq6y7N',
    dest_path='./models/test/test_umap.bin',
    unzip=True
)