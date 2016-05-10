# from timeit_decorator import timeit
# from dsfuncs import plotting_funcs as pf
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature._canny import canny
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.transform import resize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def get_images():
    df = pd.read_csv('net_data.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    imgs = {}

    count = 0
    for img in list(df['5']):
        imgs[img[-26:-4]] = rgb2gray(imread(img)).tolist()
        count += 1
        if count == 10:
            break
    with open('net_data.json', 'w') as data_file:
        data = json.dump(imgs, data_file)

    # imgs =  {'Male': male, 'Female': female, 'Forest': forest, 'Beach': beach}
    # for k, v in imgs.iteritems():
    # imgs[k] = resize(v, (300, 300))

    return imgs

# def plot_images(imgs, save_title='imgs.png', show = False):
# 	'''
# 	Input: Dictionary of images, plotting options.
# 	Output: Plot of 4 images.
# 	'''
# 	for idx, k, v in zip(range(4), imgs.keys(), imgs.values()):
# 		sub = plt.subplot(2, 2, idx + 1)
# 		plt.setp(sub.get_yticklabels(), visible=False)
# 		plt.setp(sub.get_xticklabels(), visible=False)
# 		imshow(v)
# 		pf.plt_specs(ax = sub, title='Image of ' + k, grid_bool = False)
#
# 	plt.savefig(save_title)
# 	if show:
# 		plt.show()
#
# def grayscale(imgs, show_bool = False):
# 	'''
# 	Input: Dictionary of images, plotting option.
# 	Output: Dictionary of images, plot of 4 images.
# 	Transform the colored images to greyscale, and plot them to make sure
# 	it worked.
# 	'''
# 	gray_imgs = {}
# 	for k, v in imgs.iteritems():
# 		gray_imgs[k] = rgb2gray(v)
#
# 	if show_bool:
# 		plot_images(gray_imgs, save_title='gray_imgs.png', show = show_bool)
#
# 	return gray_imgs
#
# def apply_filter(imgs, img_filter = sobel, save_title = 'a.png', show_bool = False):
# 	'''
# 	Input: Dictionary of images, image filter, plotting options.
# 	Output: Dictionary of images, plot of 4 images.
# 	Apply the filter to the set of images and plot them to make sure it worked.
# 	'''
# 	filtered_imgs = {}
# 	for k, v in imgs.iteritems():
# 		filtered_imgs[k] = img_filter(v)
#
# 	if show_bool:
# 		plot_images(filtered_imgs, save_title=save_title, show = show_bool)
#
# 	return filtered_imgs
#
# @timeit(loops = 100)
# def time_filter(imgs, img_filter = sobel):
# 	'''
# 	Input: Dictionary of images, image filter.
# 	Output: None.
# 	Used to time different filtering methods for images.
# 	'''
#
# 	for img in imgs.values():
# 		img_filter(img)
#
# def test_canny_sigma(imgs):
# 	'''
# 	Input: Dictionary of images.
# 	Output: None
# 	Apply several different levels of sigma to the canny filter, plotting
# 	each result. Figure out which canny works best.
# 	'''
#
# 	for sig in xrange(5):
# 		filtered_imgs = {}
# 		for k, v in imgs.iteritems():
# 			filtered_imgs[k] = canny(v, sigma = sig)
#
# 		save_tit = 'sig' + str(sig) + '.png'
# 		plt.suptitle('Sigma: ' + str(sig))
# 		plot_images(filtered_imgs, save_title=save_tit, show = True)
#
# def test_denoise(imgs, denoise = denoise_bilateral, save_title = 'Hooplah.png', show_bool=False):
# 	'''
# 	Input: Dictionary of images, denoise function, plotting options.
# 	Output: Plot of images with denoise applied.
# 	Apply the denoise function to the gray-scaled, canny-filtered images and examine the plots
# 	to see which denoise method we might want to use.
# 	'''
#
#
# 	denoised_imgs = {}
# 	if denoise == denoise_bilateral:
# 		for x in np.arange(start=0, stop=1, step=0.2):
# 			for k, v in imgs.iteritems():
# 				filtered_img = canny(v, sigma=1)
# 				denoised_imgs[k] = denoise(filtered_img, sigma_range=x)
#
# 			if show_bool:
# 				tit = 'Sig=' + str(x) + ':' + save_title
# 				plt.suptitle(tit)
# 				plot_images(denoised_imgs, save_title = tit, show = False)
#
# 	elif denoise == denoise_tv_chambolle:
# 		for x in np.arange(start=0.3, stop=3, step=0.54):
# 			for k, v in imgs.iteritems():
# 				filtered_img = canny(v, sigma=1)
# 				denoised_imgs[k] = denoise(filtered_img, weight=x)
#
# 			if show_bool:
# 				tit = 'Weight=' + str(x) + ':' + save_title
# 				plt.suptitle(tit)
# 				plot_images(denoised_imgs, save_title = tit, show = False)
#
# 	return denoised_imgs
#
# def split_imgs(color_imgs, gray_imgs):
# 	'''
# 	Input: Dictionary of Images, Dictionary of Images
# 	Ouput: Dictionary of Images, Dictionary of Images
#
# 	Filter the color_imgs and gray_imgs down from the 4 that they contain to the two
# 	gray and color images we actually want to use.
# 	'''
#
# 	col_imgs = {}
# 	gry_imgs = {}
#
# 	col_imgs['Forest'] = color_imgs['Forest']
# 	col_imgs['Beach'] = color_imgs['Beach']
# 	gry_imgs['Male'] = gray_imgs['Male']
# 	gry_imgs['Female'] = gray_imgs['Female']
#
# 	return col_imgs, gry_imgs
#
# def apply_KMeans(color_imgs):
# 	'''
# 	Input: Dictionary of Color Images
# 	Output: Clusters of colors per image.
# 	Fit KMeans to each class of images (here only 2) to get clusters of colors.
# 	'''
# 	clusters = []
# 	for img in color_imgs.values():
# 		nrow, ncol, depth = img.shape
# 		lst_of_pixels = [img[irow][icol] for irow in range(nrow) for icol in range(ncol)]
# 		X = np.array(lst_of_pixels)
#
# 		sklearn_km = KMeans(n_clusters=3)
# 		result = sklearn_km.fit_predict(X)
# 		clusters.append(sklearn_km.cluster_centers_)
#
# 	return clusters
#
# def featurize(grey_imgs):
# 	'''
# 	Input: Dictionary of images.
# 	Output: 2-D NP Array.
# 	Featurize the gray-scaled images and stack the results into a numpy array.
# 	'''
#
# 	imgs = [np.ravel(img) for img in grey_imgs.values()]
# 	imgs_stacked = np.r_['0', imgs]
#
# 	return imgs_stacked
#
#
#
# if __name__ == '__main__':
# 	imgs = get_images()
# 	plot_images(imgs)
#
# 	'''
# 	3.) I would choose to greyscale the male/female set of images. The color from the
# 	beach/forest pictures will be pretty informative in deciding which is which, whereas
# 	this is much less true in the male/female images. There isn't really anything inherent
# 	with color that will help you in deciding whether somebody is male or female.
# 	'''
#
# 	gray_imgs = grayscale(imgs)
# 	# sobel_imgs = apply_filter(gray_imgs, img_filter = sobel, save_title='sobel_imgs.png', show_bool = False)
# 	# canny_imgs = apply_filter(gray_imgs, img_filter = canny, save_title='canny_imgs.png', show_bool = False)
#
# 	'''
# 	4.) The canny filter performs better in terms of edge detection. It can be useful to
# 	use an edge detection filter in image classification to help speed up the process of
# 	training your model and to potentially help reduce overfitting. Since you are reducing
# 	the amount of information fed into the model to train it, it will train faster and potentially
# 	avoid some part of the overfitting that would have occured if you hadn't performed
# 	edge filtering.
# 	The canny filter takes much longer on average (0.43 seconds for the sobel and 4.9 for the canny).
# 	And to think this was only for four images...
# 	Using a sigma of 1 or 2 looks like it would work the best. Sigma of 0 picks up too many edges, and
# 	3 and 4 don't pick up enough. I'm going to go with a sigma of 1 so that we can still pick up Brad
# 	Pitts beard. It's a nice beard.
# 	'''
#
# 	# time_filter(gray_imgs, img_filter = sobel)
# 	# time_filter(gray_imgs, img_filter = canny)
# 	# test_canny_sigma(gray_imgs)
#
# 	# test_denoise(gray_imgs, denoise = denoise_tv_chambolle, save_title='denoise_chambolle.png', show_bool = True)
# 	# test_denoise(gray_imgs, denoise = denoise_bilateral, save_title='denoise_bilateral.png', show_bool = True)
#
# 	'''
# 	I'm going to use the tv denoise, with a sigma of 2.46.
# 	'''
# 	# Grab color and gray images so we can now attack them separately.
# 	color_imgs, gray_imgs = split_imgs(imgs, gray_imgs)
#
# 	centroids = apply_KMeans(color_imgs)
# 	imgs_featurized = featurize(gray_imgs)
