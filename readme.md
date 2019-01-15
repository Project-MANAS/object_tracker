# Object Tracker
Object tracker based on kalman fiters to track objects in a binary thresholded image

## Usage
To run the object tracker on a video file, run 
```bash
python object_tracker.py -i <path to video file> [OPTIONS]
```
or
```bash
python object_tracker.py --input=<path to video file> [OPTIONS]
```

Optional options include: 

| Short options | Long options | Description |
| ------------- | ------------ | ----------- |
| -d | --debug | Enable debug mode |
| -v | --verbose | Enable verbose output |
| -h | --help | Show usage help |
| -s | --show-info | Show object tracker info when visualizing output |

To generate test videos to test the tracker on, use:

```bash
python video_generator.py -o <output file> [OPTIONS]
```
or
```bash
python video_generator.py --outfile=<output file> [OPTIONS]
```

Optional options include:

| Short options | Long options | Description |
| ------------- | ------------ | ----------- |
| -h | --help | Show usage help |
| -x | --hres | Horizontal resolution | 
| -y | --vres | Vertical resolution |
| -r | --radius | Radius of objects generated |
| -b | --num-objects | Number of objects generated |
| -l | --loops | Number of times the set of n objects are generated. (A new set of objects are generated once all the objects in the current set leave the frame) |
| -n | --noise-factor | Variance of the uniform distribution from which noise is sampled |

*Note:* Long options that require areguments are used with an '=' sign.
eg: --noise-factor=5

## ROS wrapper usage

To run the ObjectTracker as a ROS node, build the package inside your catkin workspace and run
```bash
rosrun objecttracker tracker_node.py
```
tracker_node.py subscribes to ```CompressedImage``` messages on the ```tracking_frames``` topic and publishes ```DetectedObjectArray``` messages to the ```tracked_objects``` topic

To test tracker_node with dummy data, run:
```bash
rosrun onjecttracker video_generator_node.py
```
```video_generator_node``` publishes dummy binary thresholded data onto the ```tracking_frames``` topic