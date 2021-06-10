import os

def get_splendor_home():
    default_home = os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'))
    default_home = os.path.join(default_home, 'splendor')
    splendor_home = os.path.expanduser(os.getenv('SPLENDOR_HOME', default_home))
    return splendor_home

def make_splendor_home():
    home = get_splendor_home()
    if not os.path.exists(home):
        os.makedirs(home)
