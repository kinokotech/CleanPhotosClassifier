import os
import time
import traceback
import flickrapi
from urllib.request import urlretrieve
import sys
from retry import retry

flickr_api_key = "xxxxxxxxxxxxxxxxxxxx"
secret_key = "xxxxxxxxxxxxxxxxxxxx"

#第一引数にキーワードを入力する
keyword = sys.argv[1]


@retry()
def get_photos(url, filepath):
    urlretrieve(url, filepath)
    time.sleep(1)

if __name__ == '__main__':

    flicker = flickrapi.FlickrAPI(flickr_api_key, secret_key, format='parsed-json')
    response = flicker.photos.search(
        text=keyword,
        per_page=500,#取得するデータ数
        media='photos',#写真を指定
        sort='relevance',#関連順に取得
        safe_search=2,#不適切なコンテンツを削除
        extras='url_m,license'#取得するデータの種類を指定
    )
    photos = response['photos']

    try:
        if not os.path.exists('./image-data/' + keyword):
            os.mkdir('./image-data/' + keyword)

        for i,photo in enumerate(photos['photo']):
            url_m = photo['url_m']
            filepath = './image-data/' + keyword + '/' + str(i).zfill(4) + '.jpg'
            #filepath = './image-data/' + keyword + '/' + photo['id'] + '.jpg'
            get_photos(url_m, filepath)

    except Exception as e:
        traceback.print_exc()
