import os
import requests
import zipfile

import gdown

def download(url, destination, overwrite=False, quiet=False):
    if os.path.exists(destination):
        if overwrite:
            if not quiet:
                print('Removing existing: %s'%destination)
            os.remove(destination)
        else:
            if not quiet:
                print('%s already downloaded: %s'%(url, destination))
    if not os.path.exists(destination):
        if not quiet:
            print('Downloading %s to: %s'%(url, destination))
        if 'drive.google' in url:
            return gdown.cached_download(url, destination, quiet=quiet)
        else:
            r = requests.get(str(url), allow_redirects=True)
            open(destination, 'wb').write(r.content)
            return destination
    
    return None

def agree_to_zip_licenses(zip_path, license_name='license'):
    z = zipfile.ZipFile(zip_path)
    for name in z.namelist():
        if license_name in name.lower():
            print('Found License: %s'%name)
            f = z.open(name)
            print(f.read().decode('utf-8'))
            print('Agree? (y/n)')
            yn = input()
            if yn[0] not in 'Yy':
                return False
    
    return True
