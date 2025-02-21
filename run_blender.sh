export CAD_PATH=$PWD/Data/real_data/cup1_mesh/cup1.obj    # path to a given cad model(mm)
#export RGB_PATH=$PWD/Data/my_data/rgb.png           # path to a given RGB image
#export DEPTH_PATH=$PWD/Data/my_data/depth.png       # path to a given depth map(mm)
#export CAMERA_PATH=$PWD/Data/my_data/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$PWD/Data/real_data/cup1_mesh/cup1_tmp        # path to a pre-defined file for saving results
# Run instance segmentation model
export SEGMENTOR_MODEL=fastsam

cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True

#cd ../Instance_Segmentation_Model
#python run_inference_custom_mine.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH