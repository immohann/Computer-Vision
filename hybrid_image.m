% reading the baby images.
im1 = imread('baby_weird.jpg');
im2 = imread("baby_happy.jpg");

%converting image to grayscale
im1 = rgb2gray(im1);
im2 = rgb2gray(im2);

%resizing image to 512x512 dimension
im1 = imresize(im1,[512 512]);
im2 = imresize(im2,[512 512]);

%applying gaussian filter to the images
im1_blur = imgaussfilt(im1, 10, 'FilterSize',31);
im2_blur = imgaussfilt(im2, 10, 'FilterSize',31);

%fetching detail-image by subtracting the blur from resized image
im2_detail = im2 -im2_blur;

%forming hybrid image by adding image2 detail to image 1
hybrid_img = im1_blur + im2_detail;

imshow(hybrid_img);
%imshow(im2);

