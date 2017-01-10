####												----LATEST----												####
import timeit
start = timeit.default_timer()
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn import svm

def validate_ann(cnt):
	rect=cv2.minAreaRect(cnt)  
	box=cv2.boxPoints(rect) 
	box=np.int0(box)
	output=False
	width=rect[1][0]
	height=rect[1][1]
	if ((width!=0) & (height!=0)):
		if (((height/width>1.12) & (height>width)) | ((width/height>1.12) & (width>height))):
			if((height*width<1700) & (height*width>100)):
				print str(width)+' '+str(height)
				if((max(width, height)<64) & (max(width, height)>35)):
					output=True
	return output 

def showfig(image, ucmap):
    if len(image.shape)==3 :
        b,g,r = cv2.split(image)       
        image = cv2.merge([r,g,b])     
    imgplot=plt.imshow(image, ucmap)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    #plt.show()

def validate(cnt):    
    rect=cv2.minAreaRect(cnt)  
    box=cv2.boxPoints(rect)
    box=np.int0(box)  
    output=False
    width=rect[1][0]
    height=rect[1][1]
    if ((width!=0) & (height!=0)):
        if (((height/width>3) & (height>width)) | ((width/height>3) & (width>height))):
            if((height*width<16000) & (height*width>3000)):
                output=True
    return output

def generate_seeds(centre, width, height):
    minsize=int(min(width, height))
    seed=[None]*10
    for i in range(10):
        random_integer1=np.random.randint(1000)
        random_integer2=np.random.randint(1000)
        seed[i]=(centre[0]+random_integer1%int(minsize/2)-int(minsize/2),centre[1]+random_integer2%int(minsize/2)-int(minsize/2))
    return seed

def generate_mask(image, seed_point):
    h=carsample.shape[0]
    w=carsample.shape[1]

    mask=np.zeros((h+2, w+2), np.uint8)

    lodiff=50
    updiff=50
    connectivity=4
    newmaskval=255
    flags=connectivity+(newmaskval<<8)+cv2.FLOODFILL_FIXED_RANGE+cv2.FLOODFILL_MASK_ONLY
    _=cv2.floodFill(image, mask, seed_point, (255, 0, 0),
                    (lodiff, lodiff, lodiff), (updiff, updiff, updiff), flags)
    return mask

def rmsdiff(im1, im2):
    diff=im1-im2
    output=False
    if np.sum(abs(diff))/float(min(np.sum(im1), np.sum(im2)))<0.01:
        output=True
    return output
 
plt.rcParams['figure.figsize'] = 10, 10
 

plt.title('Sample Car')
#5 15
print
print
print "Enter the image name-:"
image_path=raw_input()
carsample=cv2.imread(image_path)
carsample1=cv2.imread(image_path)

plt.rcParams['figure.figsize'] = 7,7


gray_carsample=cv2.cvtColor(carsample, cv2.COLOR_BGR2GRAY)

blur=cv2.GaussianBlur(gray_carsample,(5,5),0)
#showfig(blur, plt.get_cmap('gray'))
#plt.show()

sobelx=cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
#showfig(sobelx, plt.get_cmap('gray'))
#plt.show(); 

_,th2=cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#showfig(th2, plt.get_cmap('gray'))
#plt.show()
se=cv2.getStructuringElement(cv2.MORPH_RECT,(23,2))
closing=cv2.morphologyEx(th2, cv2.MORPH_CLOSE, se)
#showfig(closing, plt.get_cmap('gray'))
#plt.show()
ha1,contours,ha2=cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#print len(contours)
#showfig(carsample, None)
 

for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt)  
        box=cv2.boxPoints(rect)
        box=np.int0(box)  
        cv2.drawContours(carsample, [box], 0, (0,0,255),2)
        break;
carsample_mask=cv2.imread(image_path)
#showfig(carsample,None)
#plt.show()

mask_list=[]
for cnt in contours:
    if validate(cnt):
        rect=cv2.minAreaRect(cnt)
        centre=(int(rect[0][0]), int(rect[0][1]))
        width=rect[1][0]
        height=rect[1][1]
        seeds=generate_seeds(centre, width, height)
        for seed in seeds:
            #cv2.circle(carsample, seed, 1, (0,0,255), -1)
            mask=generate_mask(carsample_mask, seed)
            mask_list.append(mask)  
validated_masklist=[]
#showfig(carsample, None)
#plt.show()
for mask in mask_list:
    contour=np.argwhere(mask.transpose()==255)
    if validate(contour):
        validated_masklist.append(mask)
try:
	assert (len(validated_masklist)!=0)
	connectivity=4
	newmaskval=255
	h=validated_masklist[0].shape[0]
	w=validated_masklist[0].shape[1]
	mask=np.zeros((h+2, w+2), np.uint8)
	flags=connectivity+(newmaskval<<8)+cv2.FLOODFILL_FIXED_RANGE
	_=cv2.floodFill(validated_masklist[0], mask, (0,0),255,1, 1, flags)  
	for i in range(len(validated_masklist[0])):
		for j in range(len(validated_masklist[0][i])):
			validated_masklist[0][i][j]=255-validated_masklist[0][i][j]
	#showfig(validated_masklist[0],plt.get_cmap('gray'))
	#plt.show()
	#showfig(mask,plt.get_cmap('gray'))
	ha1,contours,ha2=cv2.findContours(validated_masklist[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	#print len(contours)
	max_h=0
	rects = [cv2.boundingRect(ctr) for ctr in contours]
	for rect in rects:
		max_h=max(max_h,rect[3])
	i=0
	rects.sort()
	seg_chars=[]
	
	for rect in rects:
		if((rect[3]-max_h)<=2|(rect[3]-max_h)>=-2):
			#cv2.rectangle(carsample, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
			crop_img=carsample1[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
			crop_imgg=cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
			res = cv2.resize(crop_imgg,(8,8), interpolation = cv2.INTER_AREA)
			_,th2=cv2.threshold(res, 0,1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			horz_hist=np.sum(th2==1, axis=0)
			vert_hist=np.sum(th2==1, axis=1)
			sample=th2.flatten()
			#concatenate these features together
			feature=np.concatenate([horz_hist, vert_hist, sample])
			seg_chars.append(feature)
			#print crop_img.shape
			st='6ak'+str(i)+'.jpg'
			cv2.imwrite(st,crop_img)
			i=i+1
	#cv2.imwrite('char_contour.jpg',carsample)
	ln=len(seg_chars)
	try:
		assert (ln==10)
		#training start
		d=[]
		perfeature_mat1=[]
		Y1=['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0',
		'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0',
		'1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1',
		'1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1','1',
		'2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2',
		'2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2','2',
		'3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3',
		'3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3','3',
		'4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4',
		'4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4','4',
		'5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5',
		'5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5','5',
		'6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6',
		'6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6','6',
		'7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7',
		'7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7','7',
		'8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8',
		'8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8','8',
		'9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9',
		'9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9','9']
	


		perfeature_mat=[]
		Y=['A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A','A'
		,'B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B','B'
		,'C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C'
		,'D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D','D'
		,'E','E','E','E','E','E','E','E','E','E','E','E','E','E','E','E','E','E','E','E'
		,'F','F','F','F','F','F','F','F','F','F','F','F','F','F','F','F','F','F','F','F'
		,'G','G','G','G','G','G','G','G','G','G','G','G','G','G','G','G','G','G','G','G'
		,'H','H','H','H','H','H','H','H','H','H','H','H','H','H','H','H','H','H','H','H'
		,'I','I','I','I','I','I','I','I','I','I','I','I','I','I','I','I','I','I','I','I'
		,'J','J','J','J','J','J','J','J','J','J','J','J','J','J','J','J','J','J','J','J'
		,'K','K','K','K','K','K','K','K','K','K','K','K','K','K','K','K','K','K','K','K'
		,'L','L','L','L','L','L','L','L','L','L','L','L','L','L','L','L','L','L','L','L'
		,'M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M','M'
		,'N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N','N'
		,'O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O'
		,'P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P','P'
		,'Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q','Q'
		,'R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R','R'
		,'S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S','S'
		,'T','T','T','T','T','T','T','T','T','T','T','T','T','T','T','T','T','T','T','T'
		,'U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U','U'
		,'V','V','V','V','V','V','V','V','V','V','V','V','V','V','V','V','V','V','V','V'
		,'W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W','W'
		,'X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X','X'
		,'Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y','Y'
		,'Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z','Z']
		for i in range(520):
			im='newdata/'+str(i)+'.jpg'
			res = cv2.imread(im,0)
			_,th2=cv2.threshold(res, 0,1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			horz_hist=np.sum(th2==1, axis=0)
			vert_hist=np.sum(th2==1, axis=1)
			sample=th2.flatten()
			#concatenate these features together
			feature=np.concatenate([horz_hist, vert_hist, sample])
			perfeature_mat.append(feature)
		for i in range(500):
			im='num_dataset/'+str(i)+'.jpg'
			res = cv2.imread(im,0)
			_,th2=cv2.threshold(res, 0,1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			horz_hist=np.sum(th2==1, axis=0)
			vert_hist=np.sum(th2==1, axis=1)
			sample=th2.flatten()
			#print sample.shape
			feature=np.concatenate([horz_hist, vert_hist, sample])
			perfeature_mat1.append(feature)
		clf1=svm.SVC()
		clf = svm.SVC()
		clf.fit(perfeature_mat,Y)
		clf1.fit(perfeature_mat1,Y1)
		#training ends
		
		alpha=[]
		alpha.append(seg_chars[0])
		alpha.append(seg_chars[1])
		alpha.append(seg_chars[4])
		alpha.append(seg_chars[5])
		num=[]
		num.append(seg_chars[2])
		num.append(seg_chars[3])
		num.append(seg_chars[6])
		num.append(seg_chars[7])
		num.append(seg_chars[8])
		num.append(seg_chars[9])
		res=clf.predict(alpha)
		res1=clf1.predict(num)
		final=[]
		final.append(res[0])
		final.append(res[1])
		final.append(res1[0])
		final.append(res1[1])
		final.append(res[2])
		final.append(res[3])
		final.append(res1[2])
		final.append(res1[3])
		final.append(res1[4])
		final.append(res1[5])
		print final
	except AssertionError:
		print "Character couldnot be segmented"
except AssertionError:
	print "Plate couldnot be localized"
stop = timeit.default_timer()
print stop - start
