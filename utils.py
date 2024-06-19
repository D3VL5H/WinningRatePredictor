import os

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'{dir} created')
        return
    print (f'{dir} already exists')

