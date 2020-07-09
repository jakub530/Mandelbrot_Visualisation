import numpy as np
from PIL import ImageGrab
import cv2 
import time
import math
import cmath 
import matplotlib.pyplot as plt
import concurrent.futures
import colorsys

class slice:
    def __init__(self,start_x,end_x,start_y,end_y):
        self.x_s = start_x
        self.x_e = end_x
        self.y_s = start_y
        self.y_e = end_y
        
class boundary:
    def __init__(self,left,right,bottom,top):
        self.real_min = left
        self.real_max = right
        self.im_min = bottom
        self.im_max = top
        
class parameters:
    def  __init__(self,max_iter,horizontal_slices,vertical_slices,height,width):
        self.max_iter = max_iter
        self.h_slices = horizontal_slices
        self.v_slices = vertical_slices
        self.height   = height
        self.width    = width

class click_location:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
def zoom(param, zoom_ratio,x,y):
    global bound
    global final_image
    x_c = bound.real_min+(bound.real_max-bound.real_min)*x/param.width
    span_x = (bound.real_max-bound.real_min)*zoom_ratio
    bound.real_min = x_c - span_x/2
    bound.real_max = x_c + span_x/2
    y_c = bound.im_min+(bound.im_max-bound.im_min)*y/param.height
    span_y = (bound.im_max-bound.im_min)*zoom_ratio
    bound.im_min = y_c - span_y/2
    bound.im_max = y_c + span_y/2
    input_image = create_picture(bound,param)
    final_image = collect_image(input_image,param.h_slices,param.v_slices)
    final_image = post_process_1(final_image,param.max_iter,param.height,param.width)
    cv2.imshow('image',final_image)    
    
def rect_zoom(param,min_x,max_x,min_y,max_y):
    global bound
    global final_image
    x_c = bound.real_min+(bound.real_max-bound.real_min)*(max_x+min_x)/(2*param.width)
    span_x = (bound.real_max-bound.real_min)*(max_x-min_x)/param.width
    bound.real_min = x_c - span_x/2
    bound.real_max = x_c + span_x/2
    y_c = bound.im_min+(bound.im_max-bound.im_min)*(max_y+min_y)/(2*param.height)
    span_y = span_x/2
    bound.im_min = y_c - span_y/2
    bound.im_max = y_c + span_y/2
    input_image = create_picture(bound,param)
    final_image = collect_image(input_image,param.h_slices,param.v_slices)
    final_image = post_process_1(final_image,param.max_iter,param.height,param.width)
    cv2.imshow('image',final_image)  

def draw_rectangle(start_point,end_point,image):
    color = (255,255,255)
    thickness = 2
    cv2.rectangle(image, start_point, end_point, color, thickness) 
    cv2.imshow('image', image) 
    
def click(event, x, y, flags, param):
    global bound
    global click_loc
    global left_click_flag
    global final_image
    if(left_click_flag==1):
        tmp_image = final_image.copy()
        start_point = (min(click_loc.x, x),max(click_loc.y, y))
        end_point =   (max(click_loc.x, x),min(click_loc.y, y))
        draw_rectangle(start_point,end_point,tmp_image)
        
    if event == cv2.EVENT_LBUTTONDOWN:
        click_loc = click_location(x,y)
        left_click_flag = 1
        #re_zoom(param,zoom_ratio,x,y)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        zoom_ratio = 2.0
        zoom(param,zoom_ratio,x,y)
        
    if event == cv2.EVENT_LBUTTONUP:
        new_click_loc = click_location(x,y)
        min_x = min(click_loc.x, new_click_loc.x)
        max_x = max(click_loc.x, new_click_loc.x)
        min_y = min(click_loc.y, new_click_loc.y)
        max_y = max(click_loc.y, new_click_loc.y)
        rect_zoom(param,min_x,max_x,min_y,max_y)
        left_click_flag = 0

def create_part_picture(slice,bounds,height,width,max_iter):
    new_height = int(slice.y_e-slice.y_s)
    new_width = int(slice.x_e-slice.x_s)
    c_matrix = np.zeros((new_height,new_width), dtype=complex)
    completed_matrix = np.ones((new_height,new_width), np.uint8)
    result = np.zeros((new_height,new_width), np.uint8)
    
    for y in range(int(slice.y_s), int(slice.y_e)):
        for x in range(int(slice.x_s), int(slice.x_e)):
            c_matrix[int(y-slice.y_s)][int(x-slice.x_s)] = complex(bounds.real_min + (x / width)  * (bounds.real_max - bounds.real_min),
                                                                   bounds.im_min   + (y / height) * (bounds.im_max   - bounds.im_min))
    z_matrix = np.zeros((new_height,new_width), np.uint8)
    iter = 0
    while iter < max_iter:
        z_matrix=z_matrix*z_matrix+c_matrix
        iter += 1
        mask = abs(z_matrix*completed_matrix) > 2
        result[mask] = iter
        completed_matrix[mask] = 0
        c_matrix[mask] = 0
        z_matrix[mask] = 0
    result = result + completed_matrix*max_iter
    return result

def create_picture(bound_elem,param):
    height            = param.height
    width             = param.width
    max_iter          = param.max_iter
    horizontal_slices = param.h_slices
    vertical_slices   = param.v_slices
    heights= np.full(horizontal_slices*vertical_slices,height)
    widths= np.full(horizontal_slices*vertical_slices,width)
    bounds = np.full(horizontal_slices*vertical_slices,bound_elem)
    max_iters = np.full(horizontal_slices*vertical_slices,max_iter)
    blank_image = np.zeros((height,width), np.uint8)
    slices = []
    for y in range(vertical_slices):
        for x in range(horizontal_slices):
            slice_elem = slice(width*x/horizontal_slices,width*(x+1)/horizontal_slices,height*y/vertical_slices,height*(y+1)/vertical_slices)
            slices.append(slice_elem)
    output = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(create_part_picture, slices,bounds,heights,widths,max_iters)
        for item in results:
            output.append(item)  
    return  output

def test_threading():
    horizontal_slices = 1
    height = 720
    width = 720
    w = range(1,240)
    Time = []
    Thread = []
    
    for item in w:
        if(height%item==0):
            last_time = time.time() 
            blank_image = create_picture(boundary(-1,1,-1,1),max_iter,horizontal_slices,item,height,width)
            print('Loop took {} seconds'.format(time.time()-last_time))
            Time.append(time.time()-last_time)
            Thread.append(item)
            print('It was w' + str(item))
    plt.plot(Thread,Time)
    plt.show()
    
def collect_image(input_image,horizontal_slices,vertical_slices):
    temp_results=[]
    for y in range(vertical_slices):
        for x in range(horizontal_slices):
            if (x==0):
                temp_result = input_image[0+y*horizontal_slices]
            else:
                temp_result = np.concatenate((temp_result,input_image[x+y*horizontal_slices]),axis=1)
        temp_results.append(temp_result)
    raw_data = np.concatenate((temp_results),axis=0)   
    return raw_data

def linear_interpolation(color1, color2, t):
    return color1 * (1 - t) + color2 * t 
    
def post_process(image):
    print(1)
    
def post_process_1(image,max_iter,height,width):
    multiplier = max_iter/255
    hue = (image/multiplier).astype(np.uint8)
    
    saturation = np.full((height,width),255,np.uint8)
    value = np.full((height,width),255, np.uint8)
    mask = image > (max_iter - 1)
    value[mask] = 0
    
    output_image = np.zeros((height,width,3), np.uint8)
    output_image[:,:,0]=hue
    output_image[:,:,1]=saturation
    output_image[:,:,2]=value
    output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2BGR)
    #cv2.imshow('image',output_image)
    return output_image
    #print(output_image)
    
    
    
def post_process_histogram(image):
    MAX_ITER = 30
    unique,counts = np.unique(image,return_counts = True)
    total = sum(counts[:-1])
    hues = []
    h=0
    for i in range(MAX_ITER):
        h += histogram[i] / total
        hues.append(h)
    hues.append(h)

if __name__ == '__main__':
    max_iter = 255
    horizontal_slices = 10
    vertical_slices = 10
    height = 400
    width = 800
    left_click_flag = 0
    params = parameters(max_iter,horizontal_slices,vertical_slices,height,width)
    bound = boundary(-2,2,-1,1)
    input_image = create_picture(bound,params)
    
    final_image = collect_image(input_image,params.h_slices,params.v_slices)
    final_image = post_process_1(final_image,max_iter,height,width)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click,param=params)
    cv2.imshow('image',final_image)
    
    
    while(True):
        if cv2.waitKey(25) & 0xFF ==ord('q'):
            cv2.destroyAllWindows()
            break
            break