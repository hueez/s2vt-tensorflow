import os

root_dir = "/hdd4/lsmdc2017/"
#movie_list = [name for name in os.listdir(root_dir) if (os.path.isdir(os.path.join(root_dir, name)) and name[0].isdigit())]
movie_list = ['0002_As_Good_As_It_Gets', '0005_Chinatown', '0007_DIE_NACHT_DES_JAEGERS', '0008_Fargo']

num_movies = len(movie_list)
print "Total %d movies\n" % num_movies

jpg_dir = "/hdd4/lsmdc2017/jpgs/"
if not os.path.exists(jpg_dir):
    os.makedirs(jpg_dir)

for movie in movie_list:
    print "Processing %s" % movie
    movie_sub_dir = os.path.join(root_dir, movie)
    jpg_sub_dir = os.path.join(jpg_dir, movie)
    if not os.path.exists(jpg_sub_dir):
        os.makedirs(jpg_sub_dir)

    print jpg_sub_dir
    clip_list = [file for file in os.listdir(movie_sub_dir) if (os.path.isfile(os.path.join(movie_sub_dir, file)) and file.lower().endswith(('.mp4', '.avi')))]

    for clip in clip_list:
        clip_file = os.path.join(movie_sub_dir, clip)
        jpg_file = os.path.join(jpg_sub_dir, clip)
        os.makedirs(os.path.join(jpg_sub_dir, jpg_file[:jpg_file.rfind('.')]))
        cmd = ['ffmpeg -i ' + clip_file + ' -r 24 -an -threads 0 -f image2 ' + jpg_file[:jpg_file.rfind('.')] + "/" + jpg_file[jpg_file.rfind('/')+1:jpg_file.rfind('.')] + '_fn_%04d.jpg']
        os.system(cmd[0])
