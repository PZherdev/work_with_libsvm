import asyncio
import os
import shutil

import pandas as pd


async def make_dirs(root_dir):
    offers = pd.Series([x[10:13] for x in os.listdir(root_dir) if '.libsvm' in x]).unique()
    for offer in offers:
        dir_name = f"Offer_{offer}"
        dir_path = f"{root_dir}/{dir_name}"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f'{dir_path}. Created')
    files = [x for x in os.listdir(root_dir) if '.libsvm' in x]
    if len(files) == 0:
        print('There are no libsvm files')
        return
    for file in files:
        file_path = f"{root_dir}/{file}"
        dir_path = f"{root_dir}/Offer_{file[10:13]}"
        shutil.move(file_path, dir_path)
    print(f"Moved {len(files)} files")


async def concatenate_libsvms(root_dir):
    dirs = [x for x in os.listdir(root_dir) if 'Offer' in x]
    for diry in dirs:
        files = [f for f in os.listdir(f"{root_dir}/{diry}") if '.libsvm' in f]
        with open(f'/opt/notebooks/csv/Learning Data/LibSVM/{diry}.libsvm', 'wb') as wfd:
            for file in files:
                with open(f"{root_dir}/{diry}/{file}", 'rb') as fd:
                    shutil.copyfileobj(fd, wfd)


async def main():
    dir_name = '/opt/ml'
    dirs_making_task = asyncio.create_task(make_dirs(dir_name))
    asyncio.wait_for(await dirs_making_task, 60)
    concatenate_files_task = asyncio.create_task(concatenate_libsvms(dir_name))
    await dirs_making_task, concatenate_files_task


asyncio.run(main())
