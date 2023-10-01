from bing_image_downloader import downloader

query_string = "Odocoileus virginianus"

downloader.download(query_string, limit=20,  output_dir='deer', 
adult_filter_off=True, force_replace=False, timeout=60,filter="photo")