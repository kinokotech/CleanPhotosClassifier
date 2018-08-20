import os
import time
import traceback
import flickrapi
from urllib.request import urlretrieve
from retry import retry


class ImageCollector:

    def __init__(self,
                 flickr_api_key,
                 secret_key):
        self.flickr_api_key = flickr_api_key
        self.secret_key = secret_key
        self.photos = []
        self.keyword = ""

    def request_photos(self,
                       keyword,
                       per_page=500,
                       media='photos',
                       sort='relevance',
                       safe_search=2,
                       extras='url_m,license'):
        """
        :param keyword:
        :param per_page: 取得するデータ数
        :param media: 写真を指定
        :param sort: 関連順に取得
        :param safe_search: 不適切なコンテンツを削除
        :param extras: 取得するデータの種類を指定
        :return:
        """

        self.keyword = keyword
        flicker = flickrapi.FlickrAPI(flickr_api_key, secret_key, format='parsed-json')
        response = flicker.photos.search(
            text=keyword,
            per_page=per_page,
            media=media,
            sort=sort,
            safe_search=safe_search,
            extras=extras
        )

        self.photos = response['photos']['photo']

        return self

    def save_photos(self, output_dir, keyword=None):

        if keyword is None:
            keyword = self.keyword

        os.makedirs(output_dir + '/' + keyword, exist_ok=True)

        try:
            for i, photo in enumerate(self.photos):

                if 'url_m' not in photo:
                    continue

                url_m = photo['url_m']
                filepath = output_dir + '/' + keyword + '/' + str(i).zfill(4) + '.jpg'
                self.__get_photo(url_m, filepath)
        except:
            traceback.print_exc()

    @staticmethod
    @retry()
    def __get_photo(url, filepath):
        urlretrieve(url, filepath)
        time.sleep(1)


if __name__ == '__main__':

    """
    flickr_api_key = "xxxxxxxxxxxxxxxxxxxx"
    secret_key = "xxxxxxxxxxxxxxxxxxxx"
    """
    import api_key
    flickr_api_key = api_key.api_key
    secret_key = api_key.secret

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-k","--keyword", type=str, required=True, help="A keyword for search image on flickr!")
    p.add_argument("-o", "--output_dir", type=str, required=True, help="A path of output directory")

    args = p.parse_args()

    keyword = args.keyword
    output_dir = args.output_dir

    collector = ImageCollector(flickr_api_key, api_key)
    collector.request_photos(keyword, per_page=10).save_photos(output_dir)