
##### SAMPLE CODE FUNCTIONS USED AND CREATED IN THIS RESEARCH ####
##### FULL CODE AND EXAMPLES WILL BE AVAILABLE AT ######
### https://github.com/uceshaf/Contextual-Overviews-and-Multi-look-Inferencing ###





import rasterio as rio
from rasterio.windows import bounds, from_bounds, Window, transform
from rasterio.merge import merge
import fiona
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, rotate
import numpy as np
from numpy import load, array_equal, array, expand_dims, asarray, expand_dims, savez_compressed
import random
import glob
import fiona
from fiona.crs import from_epsg
from pyproj import Proj, transform


import os
from itertools import product, chain
from rasterio.windows import bounds, from_bounds, Window, transform
from rasterio.features import shapes

from keras import backend as K
from keras.optimizers import SGD, Adam
# Segmentation Models Library https://github.com/qubvel/segmentation_models
from segmentation_models import Unet
from segmentation_models.losses import binary_focal_dice_loss,categorical_focal_dice_loss 
from segmentation_models.metrics import iou_score,IOUScore, FScore



########################################## HELPER FUNCTIONS ################################################
############################################################################################################


def printv(*inp):
    """ Selective printing of Info based on global variable globalverbose  """
    if globalverbose==1:
        print(*inp)
        
############################################################################################################
   
def denormalize(x, percentile1=0, percentile2=100):
    """ Data normalization using percentiles and to Scale image to range 0..1  """
    x_min = np.percentile(x, percentile1)  
    x_max = np.percentile(x, percentile2)
    x_range = (x_max - x_min)+0.00001 # epsilon value to skip divide by zero if percentile is set high      
    x = (x - x_min) / (x_range).astype('float32')
    x = x.clip(0, 1).astype('float32')
    return x

############################################################################################################

def pred(arr, loaded_model, rescale_bit=0, normalize=True, p1=1, p2=98):
    """ Sematic segmentation predictions from the tile using rescaling and/or normalized bit """
    # moving channels to last for model prediction
    if arr.shape[0] < arr.shape[-1]:
        arr=np.moveaxis(arr, 0, -1)
    # normalize values using percentiles   
    if normalize==True:
        arr=denormalize(arr, p1, p2)
    # divide by a rescale_bit value to rescale to a range
    elif rescale_bit>0:
        arr = np.array(arr, np.float32) / rescale_bit
    # prediction from the model loaded already
    pred_arr    = loaded_model.predict(np.expand_dims(arr, axis=0))
    # return predictions
    return np.argmax(pred_arr, axis=-1).astype(np.uint8)

############################################################################################################

def rand_zoom(src, x,y, tile_width, tile_height, r_zoom, verbose=0):       
    """ For reading a zoomed in or zoomed out of version of an image tile,
    zoom value provided as number of pixels inward(negative) or outward (positive)"""
    # creating a rasterio window based on the r_zoom
    printv(f"########## Using a zoomed version with zoom pad of :{r_zoom} #############")
    zcol_off, zrow_off = x-r_zoom, y-r_zoom
    zwidth, zheight = tile_width+r_zoom, tile_height+r_zoom
    zwindow =Window(col_off=zcol_off, row_off=zrow_off, width=zwidth, height=zheight)
    ztile = src.read(window=zwindow, boundless=True)
    printv("zoomed tile.shape before resize ",ztile.shape)   
    # Resizing the zoomed value to the original shape of the tile
    ztile = resize(ztile, (ztile.shape[0],tile_width, tile_height), mode="constant", order=0,
                   anti_aliasing=False, preserve_range=True).astype(ztile.dtype)
    printv("zoomed tile.shape after resize",ztile.shape)
    return ztile


############################################################################################################

def rand_flip_rot90(img, rint):
    """ For creating a random flipped or rotated tiles"""
    print("\nRand Int", rint)
    if rint == 0:
        print("Before Flipud",img.shape)
        img = rotate(img, 180, (1,2), order=1)
        print("After Flipud",img.shape)
    elif rint == 1:
        print("Before Fliplr",img.shape)
        return np.fliplr(img)     
        print("After Fliplr",img.shape)
    elif rint == 2:
        print("Before Rot90",img.shape)
        img = rotate(img, 90, (1,2), order=1)
        print("After Rot90",img.shape)
    return img


############################################################################################################

def resized_preds(tile, loaded_model, rescale_bit=0, normalize=True,p1=1, p2=98, verbose=0):
    """ To get resizing in predictions, If the trained model is has different input_size than the tiles  """          
    if tile.shape[0] < tile.shape[-1]:
        tile=np.moveaxis(tile, 0, -1)
    model_inshape = loaded_model.get_input_shape_at(0)[1:4]
    tshape=tile.shape 
    printv("model_inshape, tshape", model_inshape, tshape)
    tile_resized = resize(tile, model_inshape, mode="reflect", anti_aliasing=True,
                          preserve_range=True).astype(tile.dtype)
    #predictions on resized array
    prd_resized = pred(tile_resized, loaded_model, rescale_bit=0, normalize=True, p1=p1, p2=p2)
    printv("Original Shape of Predictions: ", prd_resized.shape)
    pshape = (prd_resized.shape[0], tshape[0], tshape[1])                    
    prd = resize(prd_resized, pshape, mode="reflect", anti_aliasing=False, 
                 preserve_range=True ).astype('uint8')
    printv("Resized Shape of Predictions: ", prd.shape)   
    return prd

############################################################################################################

def raster_to_shapefile(raster_path, shapefile_path, value=None):
    """ To create shapefiles from masks or predictions rasters/tiles  """          
    # Open the raster dataset
    with rio.open(raster_path) as src:
        # Use the `rasterio.features.shapes` function to vectorize the one values in the raster
        crs = src.crs
        transform = src.transform
        if value:
            shapes = list(rio.features.shapes(src.read(masked=True), mask=src.read() == value, transform=transform))
        else:    
            shapes = list(rio.features.shapes(src.read(masked=True), mask=src.read(), transform=transform))

    # Create a schema for the shapefile
    schema = {"geometry": "Polygon", "properties": {"value": "int"}, "crs": crs}

    # Open the shapefile for writing
    with fiona.open(shapefile_path, 'w', 'ESRI Shapefile', schema, crs=crs) as dst:
        # Write the shapes to the shapefile
        for shape, value in shapes:
            dst.write({'geometry': shape, 'properties': {'value': value}})
            
            
############################################################################################################
############################################################################################################
            

def tilesampler_masks_overviews_multipreds(raster_path,mask_path="", loaded_model="", outpath="", rescale_bit=0, 
                                          normalize=True, p1=1, p2=98, bandstuple="",
                                          num_samples=0, tile_width=256, tile_height=256, x_overlap=0, y_overlap=0, 
                                          start_index=1, tpv=["tiles","masks","predictions","vectors"], 
                                          padoverviews=[64, 128], rand_zoom_int=0, flip_int=0, 
                                          multi = ["horizontal", "vertical", "diagonal"],
                                          class_values = [1],
                                          rescale_range=[],# rescale_range=[min,max,newmax]
                                         verbose:int=0, file_format="tif"):
    """
    Help
   
    The function is a generalized and most broad implementation of an image tiler for remote sensing images, 
    The function can be used for generating all the tiles with some tile size and overlaps, random tiles, 
    tiles with overviews padded, random zooms of tiles, predictions using loaded_model from tiles, 
    upsampling and downsampling of predictions, conversion to vector shapefiles of the predictions. 
    
    The input pararmeters are:
    
    raster_path: Provide the full path of the input image raster
    
    mask_path: Given that there is already a mask corresponding to the raster, provide the full path of the mask raster
                and include "masks" in tpv list. This will create tiles based on the raster and corresponding mask. 
                Default is "" so no mask is used.
    
    loaded_model: Provide a loaded keras/tf model for generating predictions. 
                  Default is "", generates ones array and shapefile if "predictions","vectors" are in tpv list. 
                  This maybe useful for starting training data genertation from raster outline shapefiles
                  When a loaded model is provided then actual predictions are saved adn vectorized as well
  
    outpath: If not provided, the rasterpath is used for saving output tiles, predictions, vectors etc.
  
    rescale_bit: The value provided if greater than 0 is used to divide the input array before predicions, default is zero, so no rescaling by default
    
    normalize, p1, p2: Normalize the array with denormalize function using percentiles p1, p2 in the range from 0 to 1
    
    bandstuple: The bands to select from the input image, can be provided as tuple like (0,1,2,3) t select first four bands
    
    num_samples: The num_samples if > 0,  will create random tiles up to num_samples. 
                If it is zero, all the tiles are generated based on the tile size and overlaps
    
    tile_width,  tile_height: Width and Height of the output tiles, default is 256, 256
    
    x_overlap, y_overlap: The overlap between succesive tiles in x and y directions, 
                This is useful only if num_samples=0, else random samples are generated so no concept of overlaps
    

    
    start_index: The tile number to start from, can be used to generate more tiles from where last ran ended
    
    tpv: a list with any valid values from ["tiles","masks","predictions","multipredictions", "vectors"] to save in outputs.
            "masks" need a mask_path, actual predictions need a loaded model, 
            "predictions" will provide predictions from a loaded model, else numpy arrays of zeros
            "multipredictions" provides functionality for multi-look inferencing on the tiles with the looks 
            provided in multi = ["horizontal", "vertical", "diagonal"] parameter.
            For details on the concept of multi-look inferencing, see paper at ___________
            "vectors" make shapefiles from masks and/or predictions
    
    padoverviews: Default is [64, 128], Number of contextual overviews to add to the tile with the pad lengths specfied as a list.
            For contextual overviews padding, see paper at ___________
    
    rand_zoom_in: specify the number for the num_samples after which a randomly zoomed tile upto 1/3rd of tile_width, tile_height is created  
    
    flip_int: specify the number for the num_samples after which a random flipup, flip, and rotate is applied to that tile..
    
    multi = ["horizontal", "vertical", "diagonal"]: Specify the looks for multi-look inferencing, if "multipredictions" is in tpv list
    
    rescale_range: Does rescaling of the image data range using list of values [min,max,newmax]. Default is empty list []. i.e. no resacle_range of data
    
    class_values=[1], if multi-class segmentation then provide a list of values such as [1,2] 
    so that aggreagtion is done for each class value
    
    verbose:default is 0. For detailed printing of each print statement, change verbose to 1, 
    
    file_format: Provide format of the outputs. Default is "tif", can use "img" or other extensions supported by rasterio
    
    /Help
    """
    if verbose!=1:
        print("\n\nTo get detailed print statements from the function, change the value in function call".upper(), "verbose=1")        
    global globalverbose
    globalverbose = verbose

    # Getting the names of the output folders for tiles, predictions and vectors
    rastname = os.path.basename(raster_path)[:-4]
    print("\n\nProcessing raster name: ", rastname)
    # lowercasing values of tpv
    tpv=[a.lower() for a in tpv]

    if outpath=="":
        outpath = os.path.dirname(raster_path)
    for k in tpv:
        outs = os.path.join(outpath, k)
        if not os.path.exists(outs):
            os.makedirs(outs)
        print(f"Output {k} are saved at: ", outs)
                        
    # Open the raster dataset    
    with rio.open(raster_path) as src:
        # Get the bounds of the raster data
        meta = src.meta
        imgwidth, imgheight = src.width, src.height
        print("\nMetaData of Input Image", meta, "\nImage Width, Image Height", imgwidth, imgheight)
        
        if mask_path and "masks" in tpv:
            ######### Is the transform of the mask or the image required???
            maskf = rio.open(mask_path)
            metam = maskf.meta
            print("\nMetaData of Mask Image", metam, "\nMask Width, Mask Height", metam["width"], metam["height"])
            if (meta["width"] != metam["width"]) or (meta["height"] != metam["height"]):
                print("THE SHAPES OF IMAGE AND MASK DO NOT MATCH, MAY NOT ALIGN WELL..")
                print(metaimg["width"], metam["width"], metaimg["height"], metam["height"])           

        outimages = []
        outpreds = []
        outmultipreds=[]
        outpredsvects = []
        
        # If num_samples is 0 then all samples have to be created
        if num_samples==0:
            print("\n ### num_samples=0, so all the tiles will be created instead of random tilings ###\n") 
            print(f"The offsets of the tiles have an x_overlap of {x_overlap} and y_overlap of {y_overlap}")
            offsets = product(range(0, imgwidth, tile_width-x_overlap), range(0, imgheight, tile_height-y_overlap))
            offsets = [(xs,ys) for xs,ys in offsets]
            num_samples =  len(offsets)
        else:
            offsets=""
      
        i=0    
        ################### The loop #########################    
        while i < num_samples:
            j=i+start_index
            # Either Generate a random start position for the tile within the bounds of the raster
            # We are using instead boundless method to read a window with zeros values padded in the window
            if offsets:
                x,y=offsets[i]
            else:
                x = np.random.randint(0, imgwidth)
                y = np.random.randint(0, imgheight)
            # tile x,y starts have been generated either from offsets or as random integers
            
            ###########################   TILES     ######################################
            if "tiles" in tpv:
                outimg_path  = f"{outpath}/tiles/{rastname}_{x}_{y}_{j}.{file_format}"          
                # We get window and tranform before going for the random zoom or direct read so that 
                # the transform remains same after zoomed is resized
                window =Window(col_off=x, row_off=y, width=tile_width, height=tile_height)  
                transform = rio.windows.transform(window, src.transform) 
                tile = src.read(window=window, boundless=True)
                # We do not need the empty 0 tiles usually found if the raster is clipped to an extent, 
                # Some images also have zeros border around them and no nodata value defined 
                if tile.max()==0:
                    if offsets:
                        printv("\nZeros Tile: Moving to the Next Tile in Offsets")
                        i+=1
                    else:
                        printv("\nSKIPPING the export of Zeros Tile: Not adding to the loop counter")
                    continue
                    
                printv("  \n",outimg_path)
                outimages.append(outimg_path)
                
                # Now, if the number of sample matches for random zoom of random flip, it is done as below
                if (rand_zoom_int>0):
                    if ((i+1)%rand_zoom_int==0):
                        printv("making zoom of the tile")
                        r_zoom = np.random.randint(-int(tile_width/3), int(tile_width/3))
                        tile = rand_zoom(src, x,y, tile_width, tile_height, r_zoom)       
                        
                if (flip_int>0):
                    if ((i+1)%flip_int==0):
                        printv("making flip of the tile")
                        randint = np.random.randint(0, 3)
                        tile = rand_flip_rot90(tile, randint)               
                        
                # If selected bands are needed, then bandstuple is provided in parameters    
                if bandstuple:
                    if type(bandstuple) is not tuple:
                        printv("If you need bands filtering, Provide input bands list as a tuple!!!")
                    else:
                        tile= tile[bandstuple, :, :]
                        printv("Changed Shapes by filtering Bands: ", tile.shape )

                metaimg = meta.copy()
                metaimg['transform'] = transform        
                metaimg['width'], metaimg['height'] = window.width, window.height
                metaimg['nodata']=0
                metaimg['count']=tile.shape[0]

                if len(padoverviews)>0:
                    tile2 = tile
                    for pad in padoverviews: 
                        dwindow = Window(col_off=x-pad, row_off=y-pad, 
                                         width=tile_width+2*pad, height=tile_height+2*pad)
                        dtile = src.read(window=dwindow, boundless=True)
                        if (rand_zoom_int>0):
                            if ((i+1)%rand_zoom_int==0):
                                printv("###\n Adding Zoomed overviews as well ####")
                                dtile = rand_zoom(src, x-pad,y-pad, tile_width+2*pad, 
                                                  tile_height+2*pad, int(r_zoom*(tile_width+2*pad)/tile_width)) 

                        dtile = resize(dtile, (1,tile.shape[1], tile.shape[2]), mode="constant", order=0,
                                       anti_aliasing=False, preserve_range=True).astype(meta["dtype"])  
                        if (flip_int>0):
                            if ((i+1)%flip_int==0):
                                dtile = rand_flip_rot90(dtile, randint)                            
                        tile2=np.vstack((tile2,dtile))     
                        printv(f"Tile shape of {tile2.shape} after {pad} size pad overview added: ")
                    
                    meta2img = metaimg.copy()
                    meta2img["count"]=tile2.shape[0]
                    if rescale_range:
                        print(rescale_range)
                        tile2= np.clip(tile2/rescale_bit, 0, 1)*rescale_range[2].astype("uint8")            
                    with rio.open(outimg_path, "w+", **meta2img, tiled=True, compress="LZW") as outrast:
                        outrast.write(tile2)
                else:        
                    if rescale_range:
                        print(rescale_range)
                        tile= np.clip((tile-rescale_range[0])/(rescale_range[1]-rescale_range[0]), 0, 1)*rescale_range[2].astype("uint8") 
                        print(tile.min(), tile.max())
                    with rio.open(outimg_path, "w+", **metaimg, tiled=True, compress="LZW") as outrast:
                        outrast.write(tile)
                        
                printv(f"Image {j} is stored at {outimg_path}")

                
            ##############################################################################   
            if mask_path and "masks" in tpv:          
                ######### Is the transform of the mask or the image required???
                # May add reading the mask from the coordinates, fr now the mask has to have same size and shape as the raster
                mask = maskf.read(window=window, boundless=True)
                if mask.max()==0:
                    printv("\n\nZEROS MASK\n\n")

                if (rand_zoom_int>0):
                    if ((i+1)%rand_zoom_int==0):
                        mask = rand_zoom(maskf, x,y, tile_width, tile_height, r_zoom)  
                if (flip_int>0):
                    if ((i+1)%flip_int==0):
                        mask = rand_flip_rot90(mask, randint) 
                        
                metam=metaimg.copy()
                metam['count']=mask.shape[0]
                metam['dtype']='uint8'
                outm_path  = f"{outpath}/masks/{rastname}_{x}_{y}_{j}.{file_format}"
                with rio.open(outm_path, "w+", **metam, tiled=True, compress="LZW") as outpred:
                    outpred.write(mask)                    
                printv(f"Mask {j} is stored at {outm_path}")
                
            ##############################################################################   
            if "predictions" in tpv:  
                # For now we are not using the padded and randomized tiles for predictions 
                # because models have not been trained on them
                # afterwards, when we have models for padded layers, we can do tile=tile2
                tile= np.moveaxis(tile, 0, -1)
                # if there is no loaded model then save zeros as predictions
                if not loaded_model:
                    printv("\n######### THERE IS NO LOADED MODEL PROVIDED, SO SAVING ONES ARRAY AS PREDICTIONS ######## \n")
                    prd = np.ones((1,tile.shape[1], tile.shape[2])).astype("uint8")

                elif tile.shape != loaded_model.get_input_shape_at(0)[1:4]:
                    printv("\n############ PREDICTIONS are resized back before and after MODEL ###############")                    
                    prd = resized_preds(tile, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)                        
                    
                else:
                    prd = pred(tile, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                
                metapred=metaimg.copy()
                metapred['count']=prd.shape[0]
                metapred['dtype']='uint8'
                outpred_path  = f"{outpath}/predictions/{rastname}_{x}_{y}_{j}.{file_format}"
                outpreds.append(outpred_path)
                with rio.open(outpred_path, "w+", **metapred, tiled=True, compress="LZW") as outpred:
                    outpred.write(prd)
                printv(f"Prediction {j} is stored at {outpred_path}")
                
                
                
            ########################## Tiling using Predictions or Multi-Look Predictions ################################                      
            if ("predictions" in tpv) and ("multipredictions" in tpv) and (len(multi)>0):
                if (rand_zoom_int==0) and (len(padoverviews)==0):
                    total_looks=[prd]    
                    if tile.shape != loaded_model.get_input_shape_at(0)[1:4]:
                        pred2 = resized_preds
                    else:
                        pred2 = pred
                        
                    printv("CREATING MULTI-LOOK INFERENCES For", multi, "and the SUM of these")  
                    xm = int(0.5*tile_width)            
                    xmn = xm + tile_width
                    xmn_1 = xm - tile_width
                    xmhalf = int(tile_width/2)
                    ym = int(0.5*tile_height)              
                    ymn = ym + tile_height
                    ymn_1 = ym - tile_height
                    ymhalf = int(tile_height/2)
                    printv(x,y,xm,ym, xmn, ymn)
                    img_array = src.read(window=Window(x-xmhalf, y-ymhalf, 2*tile_width, 2*tile_height), boundless=True)
                    # extracting the values for horizontal, vertical and diagnol looks..                 
                    if "horizontal" in multi:
                        left     = img_array[:, xm-xmhalf:xm+xmhalf, ym:ymn]                    
                        right    = img_array[:, xm+xmhalf:xmn+xmhalf, ym:ymn]
                        pred_right  = pred2(right, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                        pred_left   = pred2(left, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                        # Extracting the horizonatal predictions from half of left and right looks
                        pred_horizontal = np.zeros((pred_right.shape), dtype='uint8')
                        pred_horizontal[:, 0:xmhalf,:]     =      pred_left[: , xmhalf:tile_width, :]
                        pred_horizontal[:, xmhalf:tile_width,:] = pred_right[:, 0:xmhalf,: ]
                        total_looks.append(pred_horizontal)
                        outhoriz = f"{outpath}/multipredictions/{rastname}_{x}_{y}_{j}_horizontal.{file_format}"
                        with rio.open(outhoriz, "w+",**metapred, compress="LZW") as rast:
                            rast.write(pred_horizontal)
                            printv("Horizontal Predictions Exported at: ", outhoriz)        

                    if "vertical" in multi:
                        up       = img_array[:, xm:xmn, ym-ymhalf:ym+ymhalf]
                        down     = img_array[:, xm:xmn, ym+ymhalf:ymn+ymhalf]
                        pred_up     = pred2(up, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                        pred_down   = pred2(down, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)  
                        # Extracting the vertical predictions from half of up and down looks    
                        pred_vertical = np.zeros((pred_up.shape), dtype='uint8')
                        pred_vertical[:, :, ymhalf:tile_height]  =  pred_down[:, :, 0:ymhalf]
                        pred_vertical[:, :, 0:ymhalf]    =          pred_up[:, :, ymhalf:tile_height]
                        total_looks.append(pred_vertical)
                        outvert = f"{outpath}/multipredictions/{rastname}_{x}_{y}_{j}_vertical.{file_format}"
                        with rio.open(outvert, "w+",**metapred, compress="LZW") as rast:
                            rast.write(pred_vertical)
                            printv("Vertical Predictions Exported at: ", outvert)

                    if "diagonal" in multi:
                        lup_diag = img_array[:, xm-xmhalf:xm+xmhalf, ym-ymhalf:ym+ymhalf]
                        rup_diag = img_array[:, xm+xmhalf:xmn+xmhalf, ym-ymhalf:ym+ymhalf]
                        ldn_diag = img_array[:, xm-xmhalf:xm+xmhalf, ym+ymhalf:ymn+ymhalf]
                        rdn_diag = img_array[:, xm+xmhalf:xmn+xmhalf, ym+ymhalf:ymn+ymhalf]  
                        pred_lup_diag = pred2(lup_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                        pred_rup_diag = pred2(rup_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                        pred_ldn_diag = pred2(ldn_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                        pred_rdn_diag = pred2(rdn_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2) 
                        # Extracting the diagonal predictions from relevant quarters of the above four predictions    
                        pred_diagonal = np.zeros((pred_lup_diag.shape), dtype='uint8')
                        pred_diagonal[:, 0:xmhalf, 0:ymhalf]  =  pred_lup_diag[:, xmhalf:tile_width, ymhalf:tile_height]
                        pred_diagonal[:, xmhalf:tile_width,  0:ymhalf]  =  pred_rup_diag[:, 0:xmhalf, ymhalf:tile_height]
                        pred_diagonal[:, 0:xmhalf,  ymhalf:tile_height]  =  pred_ldn_diag[:, xmhalf:tile_width, 0:ymhalf]
                        pred_diagonal[:, xmhalf:tile_width,  ymhalf:tile_height]  =  pred_rdn_diag[:, 0:xmhalf, 0:ymhalf]
                        total_looks.append(pred_diagonal)
                        outdiag = f"{outpath}/multipredictions/{rastname}_{x}_{y}_{j}_diagonal.{file_format}"
                        with rio.open(outdiag, "w+",**metapred, compress="LZW") as rast:
                            rast.write(pred_diagonal)
                            printv("Diagonal Predictions Exported at: ", outdiag, "\n")
    

                    for cls_val in class_values: 
                        cls_look = [np.where(k==cls_val, 1, 0) for k in total_looks]
                        pred_sum = sum(cls_look).astype("uint8")         
                        outmultipred_path  = f"{outpath}/multipredictions/{rastname}_{x}_{y}_{j}_sum_class_{cls_val}.{file_format}"
                        outmultipreds.append(outmultipred_path)

                        with rio.open(outmultipred_path, "w+", **metapred, tiled=True, compress="LZW") as outpred:
                            outpred.write(pred_sum)
                        printv(f"MultiPrediction SUM {j} is stored at {outpred_path}\n")

                else:
                    # btw padding of overviews does not chnage the indivudal tile for prediction so, 
                    print("\n\n##### CAN NOT DO Tilings on MULTIPREDICTIONS WHEN RANDOM ZOOMS ARE THERE ")
                    print("AND WHEN PADDING OF THE OVERVIEWS IS INVOLVED\n\n")
                    print("Use Function 'multilook_inferences_padded' for this purpose! \n\n")

    
                
            ############################### Shapefiles from masks or predictions #########################
            if "vectors" in tpv:
                if "predictions" in tpv:
                    outpath_preds_ref  = f"{outpath}/vectors/prediction_vectors/" 
                    if not os.path.exists(outpath_preds_ref):
                        os.makedirs(outpath_preds_ref)
                    shppath = os.path.join(outpath_preds_ref, os.path.basename(outpred_path)[:-4]+".shp")
                    # Using the function raster_to_shapefile defined in helper functions
                    _=raster_to_shapefile(outpred_path, shppath)
                    printv("Shapefile of Predictions stored at: ",shppath)
                    outpredsvects.append(shppath)
                    
                if "masks" in tpv:
                    outpath_preds_refm  = f"{outpath}/vectors/mask_vectors/"  
                    if not os.path.exists(outpath_preds_refm):
                        os.makedirs(outpath_preds_refm)
                    shppathm = os.path.join(outpath_preds_refm, os.path.basename(outm_path)[:-4]+".shp")
                    # Using the function raster_to_shapefile defined in helper functions 
                    _=raster_to_shapefile(outm_path, shppathm)
                    printv("Shapefile of Mask stored at: ",shppathm)
                    outpredsvects.append(shppathm)
            i+=1  # if the tile is valid then we move to next integer

    return outimages, outpreds, outpredsvects




############################################################################################################
############################################################################################################ 

############################################################################################################
############################################################################################################ 

def multilook_inferences_padded(input_image, loaded_model="", output="",  tile_width = 256, tile_height =256, 
                                padoverviews=[], num_bands=4, bandstuple = "",  rescale_bit=1250.0, normalize=True, 
                                p1=1, p2=98, multi:list=['horizontal', 'vertical', 'diagonal'], 
                                class_values=[1], aggreg=["Sum", "And", "Or"], 
                                prefix="", file_format="tif"):
    """ 
    Help
    
    The function is the implementation of the concept for multi-look inferencing on remote sensing images, 
    as explained in the paper at ________,
    The function can be used for center, vertical, horizontal, diaganol look predictions along with
    Sum of looks raster, Intersection of all looks ("And"), Union of all looks ("Or")

    The input pararmeters are:
    
    input_image: Provide the path of the image file on which predictions or multi-look inferencing needs to be done.
    
    loaded_model: Provide the loaded keras/tensorflow model that can be used for .predict on the tiles
    
    output:Provide the path of the output, if empty, the image path is used for output folders
    
    tile_width=256, tile_height=256: Default size is 256, 256 but can be changed depending upon model input
    
    padoverviews=[], Specify as a list if there are padded overviews to generated before inferencing
    
    num_bands=4, specify the number of bands
    
    bandstuple = "", Provide if a band selection is to be done on input as a tuple, e.g, (1,3,4,6) etc  
    
    rescale_bit=1250.0,  The value to divide he image tile for rescaling before inference
    
    normalize=True, p1=1, p2=98, If True, then tiles are normalized using denormalize function in percentiles p1, p2
    
    multi:list=['horizontal', 'vertical', 'diagonal'], provide a list of multiple looks for inferencing, "vertical", "horizontal", "diagonal"
    
    aggreg=["Sum", "And", "Or"], select aggregation functions for multi-look inferencing
    
    prefix="", prefix for the output file names
    
    class_values=[1], if multi-class segmentation then provide a list of values such as [1,2] 
    so that aggreagtion is done for each class value
    
    file_format="tif", specify the fileformat of the output as "tif", "img", "jpg"
    
    """
    print(input_image)
    if output=="":
        output=input_image[:-4]
        print("Output is at ", output)
    import math
    img = rio.open(input_image)
    meta = img.meta
    imgdata = img.read()
    print("Metadata of Input Image: ", meta)

    if bandstuple:
        if type(bandstuple) is not tuple:
            print("If you need bands filtering, Provide input bands list as a tuple!!!")
        else:
            imgdata= imgdata[bandstuple, :, :]
            print("Changed Shapes by filtering Bands: ", imgdata.shape )
            
    # Get the dimensions of the data array
    origshape = imgdata.shape
    channels, width, height = origshape[0], origshape[1], origshape[2]
    print("The shape of original input Image is: ", origshape)
    
    # copy orig widh and height into new width height in case it is already divisible by tile sizes.
    newwidth, newheight = width, height     
    ### When the image size is not a multiple of tile wisht and size then need to pad the outer areas for processing by model
    # aa + (256 - aa % 256) next integer   
    if imgdata.shape[1]%tile_width != 0:
        newwidth = imgdata.shape[1] + (tile_width - imgdata.shape[1] % tile_width)
        print("New Padded width is : ", newwidth)        
    if imgdata.shape[2]%tile_height != 0:
        newheight = imgdata.shape[2] + (tile_height - imgdata.shape[2] % tile_height)
        print("New Padded height is : ", newheight)
    print("Updated Width and Heights are: ", newwidth, newheight)  
    
    
    ### ADDING TWICE OF TILE WIDHT AND HEIGHT TO GET AN ETRA BORDER 
    ### SO THAT MULTILOOK INFERENCE DOESNOT CREATE A BAD IMAGE For left right up and down looks
    newwidth_padded = newwidth+2*tile_width
    newheight_padded = newheight+2*tile_height
    print("Padded Width and Heights are: ", newwidth_padded, newheight_padded)        

    # Add one to the steps so that the 1.5times for right and down looks returns values
    # even when we skip the first i and j
    stepsx = math.ceil(newwidth/tile_width)+1
    stepsy = math.ceil(newheight/tile_height)+1
    print("\nTotal x Steps: ", stepsx, "Total y Steps: ", stepsy)
       
    img_array = np.zeros((channels, newwidth_padded, newheight_padded), dtype=meta['dtype'])
    print("\n Img_array shape", img_array.shape)
    # NOT ZERO START as have to get a zeros borderaround for multi-look
    # for multi-look inference padding the tile sizes on left and top also
    # so below copying the image data by allowing a border on left and top
    img_array[:channels, tile_width : origshape[1] + tile_width, tile_height : origshape[2]+tile_height] = imgdata
    
    # defining the output arrays with the newwidth and newheight and not the padded ones.
    out_array_center = np.zeros((1, newwidth, newheight), dtype='uint8')
    print("Original Output Array Center Shape: ", out_array_center.shape)
  
    total_looks=[]
    if len(multi)>0:
        if "horizontal" in multi:
            out_array_horizontal = np.zeros((1, newwidth, newheight), dtype='uint8')
            print("Original Output Array Horizontal Shape: ", out_array_horizontal.shape) 

        if "vertical" in multi:
            out_array_vertical = np.zeros((1, newwidth, newheight), dtype='uint8')
            print("Original Output Array Vertical Shape: ", out_array_vertical.shape) 
            
        if "diagonal" in multi:
            out_array_diagonal = np.zeros((1, newwidth, newheight), dtype='uint8')            
            print("Original Output Array Diagonal Shape: ", out_array_diagonal.shape) 
        


    for i,stepx in enumerate(range(stepsx)):
        # skipping i =0 and j =0 so that the left and top looks don't go beyond bounds
        print(f"\n stepx {stepx} of {stepsx} stepsx")
        if i == 0:
            continue
        for j,stepy in enumerate(range(stepsy)):
            if j==0:
                continue
            #print("\ni,stepx,j,stepy",i,stepx,j,stepy )
            
            x = i*tile_width            
            xn = x + tile_width
            xn_1 = x - tile_width
            xhalf = int(tile_width/2)
            y = j*tile_height              
            yn = y + tile_height
            yn_1 = y - tile_height
            yhalf = int(tile_height/2)
            #print("x, xn",x, xn, "\n y, yn",y, yn)
            
            # extracting the values for looks.. 
            center   = img_array[:, x:xn, y:yn]            
            if len(padoverviews)>0:
                center = pad_overviews_xyindices(img_array, center,  x, xn, y, yn, padoverviews, meta=meta)

                
                
            # Don't need to waste time on empty tiles predictions..            
            if center.max()==0:
                #print("Empty Tile Skipping i,j =", i,j) to not take up time of filling predictions at empty raster places
                continue
                
                
                
            # Getting predicitions
            pred_center = pred(center, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
            out_array_center[:, xn_1:x, yn_1:y] = pred_center       
            
            if len(multi)>0:
                if "horizontal" in multi:
                    left     = img_array[:, x-xhalf:x+xhalf, y:yn]
                    right    = img_array[:, x+xhalf:xn+xhalf, y:yn]
                    
                    if len(padoverviews)>0:
                        left = pad_overviews_xyindices(img_array, left,  x-xhalf, x+xhalf, y, yn, padoverviews, meta=meta)
                        right = pad_overviews_xyindices(img_array, right,   x+xhalf, xn+xhalf, y, yn, padoverviews, meta=meta)
                    
                    pred_right  = pred(right, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                    pred_left   = pred(left, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                    # Extracting the horizonatal predictions from half of left and right looks
                    pred_horizontal = np.zeros((pred_right.shape), dtype='uint8')
                    pred_horizontal[:, 0:xhalf,:]     =      pred_left[: , xhalf:tile_width, :]
                    pred_horizontal[:, xhalf:tile_width,:] = pred_right[:, 0:xhalf,: ]
                    
                    out_array_horizontal[:, xn_1:x, yn_1:y] = pred_horizontal  
                   
                    
                if "vertical" in multi:
                    up       = img_array[:, x:xn, y-yhalf:y+yhalf]
                    down     = img_array[:, x:xn, y+yhalf:yn+yhalf]
                    if len(padoverviews)>0:
                        up = pad_overviews_xyindices(img_array, up,  x, xn, y-yhalf, y+yhalf, padoverviews, meta=meta)
                        down = pad_overviews_xyindices(img_array, down,  x, xn, y+yhalf, yn+yhalf, padoverviews, meta=meta)                    
                    
                    pred_up     = pred(up, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                    pred_down   = pred(down, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)  
                    # Extracting the vertical predictions from half of up and down looks    
                    pred_vertical = np.zeros((pred_up.shape), dtype='uint8')
                    pred_vertical[:, :, yhalf:tile_height]  =  pred_down[:, :, 0:yhalf]
                    pred_vertical[:, :, 0:yhalf]    =          pred_up[:, :, yhalf:tile_height] 
                    
                    out_array_vertical[:, xn_1:x, yn_1:y] = pred_vertical

                
                if "diagonal" in multi:
                    lup_diag = img_array[:, x-xhalf:x+xhalf, y-yhalf:y+yhalf]
                    rup_diag = img_array[:, x+xhalf:xn+xhalf, y-yhalf:y+yhalf]
                    ldn_diag = img_array[:, x-xhalf:x+xhalf, y+yhalf:yn+yhalf]
                    rdn_diag = img_array[:, x+xhalf:xn+xhalf, y+yhalf:yn+yhalf]
                    if len(padoverviews)>0:
                        lup_diag = pad_overviews_xyindices(img_array, lup_diag,  x-xhalf, x+xhalf, y-yhalf, y+yhalf, padoverviews, meta=meta)
                        rup_diag = pad_overviews_xyindices(img_array, rup_diag,  x+xhalf, xn+xhalf, y-yhalf, y+yhalf, padoverviews, meta=meta)                    
                        ldn_diag = pad_overviews_xyindices(img_array, ldn_diag,  x-xhalf, x+xhalf, y+yhalf, yn+yhalf, padoverviews, meta=meta)
                        rdn_diag = pad_overviews_xyindices(img_array, rdn_diag,  x+xhalf, xn+xhalf, y+yhalf, yn+yhalf, padoverviews, meta=meta)                    
                                                                                
                    pred_lup_diag = pred(lup_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                    pred_rup_diag = pred(rup_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                    pred_ldn_diag = pred(ldn_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2)
                    pred_rdn_diag = pred(rdn_diag, loaded_model, rescale_bit=rescale_bit, normalize=normalize, p1=p1, p2=p2) 
                    # Extracting the diagonal predictions from half of up and down looks    
                    pred_diagonal = np.zeros((pred_lup_diag.shape), dtype='uint8')
                    pred_diagonal[:, 0:xhalf, 0:yhalf]  =  pred_lup_diag[:, xhalf:tile_width, yhalf:tile_height]
                    pred_diagonal[:, xhalf:tile_width,  0:yhalf]  =  pred_rup_diag[:, 0:xhalf, yhalf:tile_height]
                    pred_diagonal[:, 0:xhalf,  yhalf:tile_height]  =  pred_ldn_diag[:, xhalf:tile_width, 0:yhalf]
                    pred_diagonal[:, xhalf:tile_width,  yhalf:tile_height]  =  pred_rdn_diag[:, 0:xhalf, 0:yhalf]
                  
                    out_array_diagonal[:, xn_1:x, yn_1:y] = pred_diagonal 
            

    meta['count']=1
    meta['nodata']=0
    meta['dtype']='uint8'
    
    # Broadcasting and writing the output arrays
    out_array_center = out_array_center[:, 0:origshape[1], 0:origshape[2]]
    print("Center Array Shape after broadcasting: ", out_array_center.shape)   
    
    total_looks.append(out_array_center)
    with rio.open(output+prefix+"_center."+file_format, "w+",**meta, compress="LZW") as rast:
        rast.write(out_array_center)
        print("Center Predictions Exported at: ", output, "\n\n")

    if len(multi)>0:
        if "horizontal" in multi:
            out_array_horizontal = out_array_horizontal[:, 0:origshape[1], 0:origshape[2]]
            print("Horizontal Array Shape after broadcasting: ", out_array_horizontal.shape)
            
            total_looks.append(out_array_horizontal)
            with rio.open(output+prefix+"_horizontal."+file_format, "w+",**meta, compress="LZW") as rast:
                rast.write(out_array_horizontal)
                print("Horizontal Predictions Exported at: ", output+"_horizontal."+file_format, "\n\n")

        if "vertical" in multi:
            out_array_vertical = out_array_vertical[:, 0:origshape[1], 0:origshape[2]]
            print("Vertical Array Shape after broadcasting: ", out_array_vertical.shape)
            
            total_looks.append(out_array_vertical)
            with rio.open(output+prefix+"_vertical."+file_format, "w+",**meta, compress="LZW") as rast:
                rast.write(out_array_vertical)
                print("Vertical Predictions Exported at: ", output+"_vertical."+file_format, "\n\n")

        if "diagonal" in multi:
            out_array_diagonal = out_array_diagonal[:, 0:origshape[1], 0:origshape[2]]
            print("Diagonal Array Shape after broadcasting: ", out_array_diagonal.shape)
            
            total_looks.append(out_array_diagonal)
            with rio.open(output+prefix+"_diagonal."+file_format, "w+",**meta, compress="LZW") as rast:
                rast.write(out_array_diagonal)
                print("Diagonal Predictions Exported at: ", output+"_diagonal."+file_format, "\n\n")


        aggreg = [a.lower() for a in aggreg]
        if "and" in aggreg:
            for cls_val in class_values:
                if ("horizontal" in multi) and ("vertical" in multi) and ("diagonal" in multi):
                    and_result = np.where((out_array_horizontal==cls_val) & (out_array_vertical==cls_val) & 
                      (out_array_diagonal==cls_val) & (out_array_center==cls_val), 1, 0).astype("uint8") 
                    print("The AND Array Shape after broadcasting: ", and_result.shape)   
                    with rio.open(output+prefix+str(cls_val)+"_AndMerge."+file_format, "w+",**meta, compress="LZW") as rast:
                        rast.write(and_result)
                        print("AND Predictions Exported at: ", output+prefix+"class_"+str(cls_val)+"_AndMerge."+file_format, "\n\n")
                else:
                    print("CAN'T SAVE AND RASTER, NEED ALL LOOKS HORIZONTAL, VERTICAL, DIAGONAL.. CALCULATE USING SUM")
        if "or" in aggreg:
            for cls_val in class_values:
                if ("horizontal" in multi) and ("vertical" in multi) and ("diagonal" in multi):
                    or_result = np.where((out_array_horizontal==cls_val) | (out_array_vertical==cls_val) | 
                              (out_array_diagonal==cls_val) | (out_array_center==cls_val), 1, 0).astype("uint8") 
                    print("The OR Array Shape after broadcasting: ", or_result.shape)   
                    with rio.open(output+prefix+str(cls_val)+"_OrMerge."+file_format, "w+",**meta, compress="LZW") as rast:
                        rast.write(or_result)
                        print("OR Predictions Exported at: ", output+prefix+"class_"+str(cls_val)+"_OrMerge."+file_format, "\n\n")
                else:
                    print("CAN'T SAVE OR RASTER, NEED ALL LOOKS HORIZONTAL, VERTICAL, DIAGONAL.. CALCULATE USING SUM")

        if "sum" in aggreg:
            for cls_val in class_values: 
                if ("horizontal" in multi) and ("vertical" in multi) and ("diagonal" in multi): 
                    sum_result = (np.where(out_array_horizontal==cls_val, 1, 0) + np.where(out_array_vertical==cls_val, 1, 0) + 
                      np.where(out_array_diagonal==cls_val, 1, 0) + np.where(out_array_center==cls_val, 1, 0)).astype("uint8") 
                    
                    
                    print("The SUM Array Shape after broadcasting: ", sum_result.shape)             
                #sum_result =sum(total_looks)
                    print("The SUM Array Shape after broadcasting: ", sum_result.shape)   
                    with rio.open(output+prefix+str(cls_val)+"_SumMerge."+file_format, "w+",**meta, compress="LZW") as rast:
                        rast.write(sum_result)
                        print("SUM Predictions Exported at: ", output+prefix+"class_"+str(cls_val)+"_SumMerge."+file_format, "\n\n")


        
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

        
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

