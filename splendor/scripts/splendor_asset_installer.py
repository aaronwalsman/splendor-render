#!/usr/bin/env python
import argparse

from splendor.home import get_splendor_home
from splendor.assets import install_assets

asset_urls = {
    'default_assets':
        #'https://drive.google.com/uc?id=1DfSVirWRjSWxfmIu2Lund_FMpHOPphZu',
        #'https://drive.google.com/uc?id=1JHtmy-nFQQj6UZbJJjxN1i42MUGT2xXb',
        #'https://drive.google.com/uc?id=1gPZw-D3zo_5DAQbpKyP2aOdCn-Wp02au',
        'https://drive.google.com/uc?id=1ANZRTqF-23r0bVatLXqJUJ4PLg9BSa9G',
}

parser = argparse.ArgumentParser(description='Splendor Asset Installer')
parser.add_argument('--asset-package', default='default_assets', type=str,
    help=('Name of the asset package to download '
        '(%s' + ',%s'*(len(asset_urls)-1) + ')')%tuple(asset_urls.keys()))
parser.add_argument('--asset-url', default=None, type=str,
    help='URL of an asset package to download')
parser.add_argument('--destination', default=get_splendor_home(), type=str,
    help='Download Location')
parser.add_argument('--overwrite', action='store_true',
    help='Overwrite any existing files and redownload')
parser.add_argument('--cleanup-zip', action='store_true',
    help='Remove the downloaded zip after decompression')

def main():
    args = parser.parse_args()

    asset_url = args.asset_url
    asset_name = args.asset_package
    if asset_url is None:
        asset_url = asset_urls[args.asset_package]

    install_assets(
        asset_url,
        asset_name,
        overwrite=args.overwrite,
        cleanup_zip=args.cleanup_zip,
    )
