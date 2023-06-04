import time
import uuid
import random
import datetime
import dearpygui.dearpygui as dpg
import paho.mqtt.client as mqtt
import threading
import numpy as np
import argparse

import common

test = False # Set to true to test dashboard with fixed data.

def save_state(state, filename="state.json"):
    import json

    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, (datetime.datetime,)):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, cls=DateTimeEncoder)

def load_state(filename="state.json"):
    import json

    class DateTimeDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

        def object_hook(self, source):
            time_format = '%Y-%m-%d %H:%M:%S'

            for k, v in source.items():
                if isinstance(v, np.integer):
                    source[k] = int(v)
                if isinstance(v, str):
                    try:
                        source[k] = datetime.datetime.strptime(v, time_format)
                    except:
                        pass
            return source

    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f, cls=DateTimeDecoder)

def now():
	time_format = '%Y-%m-%d %H:%M:%S'
	return str(datetime.datetime.strptime(time.strftime(time_format, time.localtime()), time_format))

def get_datetime_from_string(s):
	time_format = '%Y-%m-%d %H:%M:%S'
	return datetime.datetime.strptime(s, time_format)

def get_string_from_datetime(d):
	return str(d)

def pretty_print_timedelta(td):
    x = str(td).split(',')
    s = f"{x[0].split(' ')[0]}d" if len(x) == 2 else ""
    x = x[-1]
    hms = [i.lstrip("0").split('.')[0] for i in x.strip().split(":")]
    if hms == ['', '', '']:
        hms[-1] = '0'
    return s + "".join([x + y for x, y in zip(hms, ["h", "m", "s"]) if x != ''])

def get_last_ping_for_client(data, client):
	return data[client]["last_ping"]

def get_ticks_for_time_delta(time_delta):
    if time_delta.seconds == 60: # minute
        return 60
    elif time_delta.seconds == 60*60: # hour
        return 60
    elif time_delta.days == 1: # day
        return 24
    elif time_delta.days == 7: # week
        return 7
    elif time_delta.days == 30: # month
        return 30
    else: # year/all
        return 52

def set_x_axis(time_delta, axis_name):
    if time_delta.seconds == 60: # minute
        dpg.set_axis_ticks(axis_name, tuple([(str(60-i)+"s" if i % 10 == 0 else "", i) for i in range(0, 60+1, 1)]))
        dpg.set_axis_limits(axis_name, 0, 60)
    elif time_delta.seconds == 60*60: # hour
        dpg.set_axis_ticks(axis_name, tuple([(str(60-i)+"m" if i % 10 == 0 else "", i) for i in range(0, 60+1, 1)]))
        dpg.set_axis_limits(axis_name, 0, 60)
    elif time_delta.days == 1: # day
        dpg.set_axis_ticks(axis_name, tuple([(str(24-i)+"h", i) for i in range(0, 24+1, 1)]))
        dpg.set_axis_limits(axis_name, 0, 24)
    elif time_delta.days == 7: # week
        dpg.set_axis_ticks(axis_name, tuple([(str(7-i)+"d", i) for i in range(0, 7+1, 1)]))
        dpg.set_axis_limits(axis_name, 0, 7)
    elif time_delta.days == 30: # month
        dpg.set_axis_ticks(axis_name, tuple([(str(30-i)+"d", i) for i in range(0, 30+1, 1)]))
        dpg.set_axis_limits(axis_name, 0, 30)
    else: # year/all
        dpg.set_axis_ticks(axis_name, tuple([(str(52-i)+"w" if i % 2 == 0 else "", i) for i in range(0, 52+1, 1)]))
        dpg.set_axis_limits(axis_name, 0, 52)

def map_to_ticks(cur_time_delta, time_delta):
    if time_delta.seconds == 60:
        return int(cur_time_delta.total_seconds())
    elif time_delta.seconds == 60*60:
        return int(cur_time_delta.total_seconds() // 60)
    elif time_delta.days == 1:
        return int(cur_time_delta.total_seconds() // (60*60))
    elif time_delta.days == 7:
        return int(cur_time_delta.days)
    elif time_delta.days == 30:
        return int(cur_time_delta.days)
    else: # year/all
        return int(cur_time_delta.days // 7)

def get_number_of_detections_for_time_delta(detections, time_delta, cur_datetime):
    num_ticks = get_ticks_for_time_delta(time_delta)
    datay = [0 for _ in range(num_ticks + 1)]

    for d in detections:
        delta = map_to_ticks(cur_datetime - d["time"], time_delta)

        if delta < num_ticks:
            datay[num_ticks - delta] += 1

    return datay

def update_series(data, time_delta, cur_datetime):
    num_ticks = get_ticks_for_time_delta(time_delta)
    datax = [i - 0.5 for i in list(range(0, num_ticks + 1))]
    global_datay = [0 for _ in range(num_ticks + 1)]

    for i, client_name in enumerate(list(data)[:max_displayed_realtime_clients]):
        # datay = [random.randint(0, 4) for i in range(0, num_ticks + 1)]
        datay = get_number_of_detections_for_time_delta(data[client_name]['detections'], time_delta, cur_datetime)
        dpg.set_value(f"series_tag_{i}", [datax, datay])

        set_x_axis(time_delta, f"x_axis_{i}")
        dpg.set_axis_ticks(f"y_axis_{i}", tuple([(str(i), i) for i in [0, max(datay)]]))
        dpg.fit_axis_data(f"y_axis_{i}")

        for j in range(num_ticks + 1):
            global_datay[j] += datay[j]

    dpg.set_value('series_tag', [datax, global_datay])

    set_x_axis(time_delta, "x_axis")
    max_value = max(global_datay)
    dpg.set_axis_ticks("y_axis", tuple([(str(i), i) for i in range(0, max_value+1, max(1, (max_value+1)//4))]))
    dpg.fit_axis_data(f"y_axis")

def update_activity(data, time_delta, cur_datetime):
    detections = []
    for client_name in data:
        for det in data[client_name]["detections"]:
            if time_delta == datetime.timedelta.max or cur_datetime <= det["time"] + time_delta:
                detections.append((client_name, det["time"], len(det["boxes"])))

    detections = sorted(detections, key=lambda x: x[1], reverse=True)

    for i in dpg.get_item_children("activity_table")[1]:
        dpg.delete_item(i)

    num_lines = 10
    for i in range(num_lines):
        if i < len(detections):
            d = detections[i]

            with dpg.table_row(parent="activity_table"):
                dpg.add_text(f"{d[1]}")
                dpg.add_text(f"{d[0]}")
                dpg.add_text(f"{d[2]}")

def update_clients(data, time_delta, cur_datetime):
    timeout = datetime.timedelta(minutes=5)

    active = []
    inactive = []

    for client_name in data:
        if cur_datetime < data[client_name]['last_ping'] + timeout:
            active.append((client_name, cur_datetime - data[client_name]['last_ping']))
        else:
            inactive.append((client_name, cur_datetime - data[client_name]['last_ping']))

    for i in dpg.get_item_children("filter_id_active")[1]:
        dpg.delete_item(i)

    for client_name, delta_time in active:
        with dpg.group(horizontal=True, filter_key=client_name, parent="filter_id_active"):
            dpg.add_image("check_mark", width=22, height=22)
            dpg.add_text(f"{client_name} (last ping: {pretty_print_timedelta(delta_time)} ago)")

    for i in dpg.get_item_children("filter_id_inactive")[1]:
        dpg.delete_item(i)

    for client_name, delta_time in inactive:
        with dpg.group(horizontal=True, filter_key=client_name, parent="filter_id_inactive"):
            dpg.add_image("cross_mark", width=22, height=22)
            dpg.add_text(f"{client_name} (last ping: {pretty_print_timedelta(delta_time)} ago)")

def update_summary(data, time_delta, cur_datetime):
    values = [sum(get_number_of_detections_for_time_delta(data[client_name]['detections'], time_delta, cur_datetime)) for client_name in data]
    labels = [client_name for index, client_name in enumerate(data) if values[index]]
    values = list(filter(lambda x: x, values))

    for i in dpg.get_item_children("pie")[1]:
        dpg.delete_item(i)
    dpg.add_pie_series(0.6, 0.5, radius=0.5, values=values, labels=labels, parent="pie")

def update_detections(data, time_delta, cur_datetime):
    detections = []
    for client_name in data:
        for det in data[client_name]["detections"]:
            if time_delta == datetime.timedelta.max or cur_datetime <= det["time"] + time_delta:
                detections.append((client_name, det["time"], len(det["boxes"]), det["image"]))

    detections = sorted(detections, key=lambda x: x[1])[-6:]

    for i in dpg.get_item_children("detections")[1]:
        dpg.delete_item(i)

    for client_name, det_time, _, image in detections:
        with dpg.group(parent="detections"):
            dpg.add_text(client_name)
            dpg.add_text(det_time)
            dpg.add_image(image, width=250, height=250)

def update_realtime_detections(data, client_name, cur_datetime):
    last_id = data["last_id"]

    if last_id:
        for i in dpg.get_item_children("realtime_detection")[1]:
            dpg.delete_item(i)

        width = dpg.get_item_width(f"cur_texture_tag_{last_id}")
        height = dpg.get_item_height(f"cur_texture_tag_{last_id}")

        factor = 800 / height * 0.9

        width *= factor
        height *= factor

        pos = (1920/2-width/2, 10+800/2-height/2)

        with dpg.group(parent="realtime_detection"):
            dpg.add_image(f"cur_texture_tag_{last_id}", pos=pos, width=width, height=height)

        is_detecting = False

        if client_name in data:
            for det in data[client_name]["detections"]:
                if cur_datetime - det["time"] < datetime.timedelta(seconds=1):
                    is_detecting = True
                    break

        dpg.set_value("realtime_detections_background", (150, 0, 0) if is_detecting else (37, 37, 38))

def update_realtime_detections_history(data, client_name, cur_datetime):
    if client_name not in data:
        return

    detections = []
    for det in data[client_name]["detections"]:
        if cur_datetime - det["time"] < datetime.timedelta(days=1):
            detections.append((det["time"], len(det["boxes"]), det["image"]))

    detections = sorted(detections, key=lambda x: x[1], reverse=True)[-10:]

    for i in dpg.get_item_children("realtime_detections")[1]:
        dpg.delete_item(i)

    for det_time, _, image in detections:
        with dpg.group(parent="realtime_detections"):
            dpg.add_text(str(det_time).split(' ')[1], indent=8)
            dpg.add_image(image, width=175, height=175, indent=8)

def search_callback(sender, filter_string):
    dpg.set_value("filter_id_active", filter_string)
    dpg.set_value("filter_id_inactive", filter_string)

def on_connect(client, userdata, flags, rc):
    print(f"on_connect: {userdata} {flags} {rc}")

def on_message(client, userdata, msg):
    now = datetime.datetime.now()
    payload = common.decode_data(msg.payload.decode())
    print(f"on_message ({now}): {msg.topic}")
    client_id, timestamp, image, predictions = payload

    if not dashboard:
        if client_name != client_id:
            return

    if msg.topic == "DETECTION":
        v = uuid.uuid4()

        with lock:
            if client_id not in data:
                data[client_id] = {}
                data[client_id]["detections"] = []
            data[client_id]["last_ping"] = now
            num_predictions = len(predictions) // 7
            boxes = np.split(predictions, num_predictions)
            boxes = [box[1:5].astype(np.int32) for box in boxes]
            boxes = [[box[1], box[0], box[3], box[2]] for box in boxes]
            data[client_id]["detections"].append({"time": now, "image": f"texture_tag_{v}", "boxes": boxes})

            image = np.dstack((image, 255*np.ones(image.shape[:-1])))[::2,::2,:] / 255
            height, width = image.shape[0], image.shape[1]
            image = image.flatten()

            with dpg.texture_registry():
                dpg.add_static_texture(width=width, height=height, default_value=image, tag=f"texture_tag_{v}")

    if not dashboard and msg.topic == "IMAGE":
        with dpg.texture_registry():
            image = np.dstack((image, 255*np.ones(image.shape[:-1]))) / 255
            height, width = image.shape[0], image.shape[1]
            image = image.flatten()

            with lock:
                with dpg.texture_registry():
                    last_id = data["last_id"]

                    if last_id:
                        dpg.delete_item(f"cur_texture_tag_{last_id}")

                    last_id += 1            
                    dpg.add_static_texture(width=width, height=height, default_value=image, tag=f"cur_texture_tag_{last_id}")
                    data["last_id"] = last_id

dpg.create_context()

with dpg.font_registry():
    global_font = dpg.add_font("resources/arial.ttf", 16)

with dpg.texture_registry(show=False):
    width, height, channels, value = dpg.load_image("resources/274C_color.png")
    dpg.add_static_texture(width=width, height=height, default_value=value, tag="cross_mark")

    width, height, channels, value = dpg.load_image("resources/2714_color.png")
    dpg.add_static_texture(width=width, height=height, default_value=value, tag="check_mark")

    if test:
        files = [
            "test/image_0.png",
            "test/image_1.png",
            "test/image_2.png",
            "test/image_3.png",
            "test/image_4.png",
        ]

        for index, filename in enumerate(files):
            width, height, channels, value = dpg.load_image(filename)
            dpg.add_static_texture(width=width, height=height, default_value=value, tag=f"texture_tag_file_{index}")

datax = list(range(0, 50))
datay = [0]*50

data = {}

if test:
    time0 = get_datetime_from_string(now()) - datetime.timedelta(days=12, seconds=5)
    time1 = get_datetime_from_string(now()) - datetime.timedelta(days=12, seconds=12)
    time2 = get_datetime_from_string(now()) - datetime.timedelta(days=12, seconds=21)
    time3 = get_datetime_from_string(now()) - datetime.timedelta(days=2, seconds=2)
    time4 = get_datetime_from_string(now()) - datetime.timedelta(days=0, seconds=42)

    t0 = {"time": time0, "image": "texture_tag_file_0", "boxes": [[0,1,2,3]]}
    t1 = {"time": time1, "image": "texture_tag_file_1", "boxes": [[0,1,2,3]]}
    t2 = {"time": time2, "image": "texture_tag_file_2", "boxes": [[0,1,2,3]]}
    t3 = {"time": time3, "image": "texture_tag_file_3", "boxes": [[0,1,2,3]]}
    t4 = {"time": time4, "image": "texture_tag_file_4", "boxes": [[0,1,2,3]]}

    data = {
        "Client 1": {"last_ping": time0, "detections": [t0, t1, t2]},
        "Client 2": {"last_ping": time3, "detections": [t3]},
        "Client 3": {"last_ping": time4, "detections": [t4]},
    }

parser = argparse.ArgumentParser(prog="gui.py", description="Battery Spotter GUI")
parser.add_argument("--mode", default="dashboard", const="dashboard", nargs="?", choices=["dashboard", "realtime"], help="GUI mode (default: %(default)s)")
parser.add_argument("--mqtt_broker_address", "-a", type=str, default="localhost")
parser.add_argument("--mqtt_broker_port", "-p", type=int, default="1883")
parser.add_argument("--client_name", "-n", type=str)
args = parser.parse_args()

dashboard = args.mode == "dashboard"
mqtt_broker = (args.mqtt_broker_address, args.mqtt_broker_port)

if not dashboard:
    parser = argparse.ArgumentParser(prog="gui.py", description="Battery Spotter GUI")
    parser.add_argument("--mode", default="dashboard", const="dashboard", nargs="?", choices=["dashboard", "realtime"], help="GUI mode (default: %(default)s)")
    parser.add_argument("--mqtt_broker_address", "-a", type=str, default="localhost")
    parser.add_argument("--mqtt_broker_port", "-p", type=int, default="1883")
    parser.add_argument("--client_name", "-n", type=str, required=True)
    args = parser.parse_args()

    client_name = args.client_name

if not dashboard:
    data["last_id"] = 0

print(data)

if dashboard:
    max_displayed_realtime_clients = 3

    with dpg.window(label="Latest detections", pos=(0,720), width=1920-360, height=360):
        with dpg.group(horizontal=True, tag="detections"):
            pass

    with dpg.window(label="Realtime data (global)", tag="win_realtime_data_global", pos=(0,0), width=1920-360, height=360):
        with dpg.plot(label="Number of batteries detected over time", height=320, width=1500, no_mouse_pos=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag="x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Detections", tag="y_axis")
            dpg.fit_axis_data("y_axis")

            dpg.add_bar_series(datax, datay, parent="y_axis", tag="series_tag")

    with dpg.window(label="Realtime data (per client)", tag="win_realtime_data_clients", pos=(0,360), width=1920-360, height=360):
        for i in range(max_displayed_realtime_clients):
            has_client = i < len(list(data))
            with dpg.plot(label=f"Number of batteries detected over time ({list(data)[i] if has_client else ''})", height=100, width=1500, no_mouse_pos=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="Time", tag=f"x_axis_{i}")
                dpg.add_plot_axis(dpg.mvYAxis, label="Detections", tag=f"y_axis_{i}")
                dpg.set_axis_ticks(f"y_axis_{i}", tuple([(str(i), i) for i in [0]]))
                dpg.fit_axis_data(f"y_axis_{i}")

                dpg.add_bar_series(datax, datay, parent=f"y_axis_{i}", tag=f"series_tag_{i}")

    with dpg.window(label="Summary", tag="win_summary", pos=(1920-360, 0), width=360, height=360):
        with dpg.plot(label="Detections per client", height=320, width=320, no_mouse_pos=True):
            dpg.add_plot_legend(horizontal=True, outside=True, location=dpg.mvPlot_Location_South)
            dpg.add_plot_axis(dpg.mvXAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True)
            dpg.set_axis_limits(dpg.last_item(), 0, 1.2)
            with dpg.plot_axis(dpg.mvYAxis, label="", no_gridlines=True, no_tick_marks=True, no_tick_labels=True, tag="pie"):
                pass

    with dpg.window(label="Clients", tag="win_clients", pos=(1920-360, 360), width=360, height=360):
        with dpg.group(horizontal=True):
            dpg.add_text("Filter: ")
            dpg.add_input_text(callback=search_callback, hint="type to filter client names")

        dpg.add_separator()

        dpg.add_text(f"Active:")
        with dpg.group(tag="active_clients"):
            with dpg.filter_set(id="filter_id_active"):
                pass

        dpg.add_text(f"Inactive:")
        with dpg.group(tag="inactive_clients"):
            with dpg.filter_set(id="filter_id_inactive"):
                pass

    def combo_callback(sender, app_data):
        global time_delta

        if app_data == "minute":
            time_delta = datetime.timedelta(minutes=1)
        if app_data == "hour":
            time_delta = datetime.timedelta(hours=1)
        if app_data == "day":
            time_delta = datetime.timedelta(days=1)
        elif app_data == "week":
            time_delta = datetime.timedelta(weeks=1)
        elif app_data == "month":
            time_delta = datetime.timedelta(days=30)
        elif app_data == "year":
            time_delta = datetime.timedelta(days=365)
        elif app_data == "all":
            time_delta = datetime.timedelta.max

    with dpg.window(label="Activity", tag="win_activity2", pos=(1920-360, 360+360), width=360, height=360):
        with dpg.group(horizontal=True):
            dpg.add_text("Time period:")
            dpg.add_combo(items=["minute","hour","day","week","month","year","all"], default_value="day", width=70, callback=combo_callback)

        with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                       borders_innerV=True, borders_outerV=True, row_background=True, tag="activity_table", width=330, resizable=False):
            dpg.add_table_column(label="Time", width_fixed=True)
            dpg.add_table_column(label="Client", width_fixed=True)
            dpg.add_table_column(label="Detections", width_fixed=True)

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (37, 37, 38), category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvPlotStyleVar_PlotBorderSize, 0, category=dpg.mvThemeCat_Plots)

if not dashboard:
    with dpg.window(label="Realtime View", width=1920, height=800) as win:
        with dpg.group(tag="realtime_detection"):
            pass

        with dpg.theme() as container_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (37, 37, 38), category=dpg.mvThemeCat_Core, tag="realtime_detections_background")
        dpg.bind_item_theme(win, container_theme)

    with dpg.window(label="Realtime View History", pos=(0, 800), width=1920, height=1080-800) as win:
        with dpg.group(horizontal=True, tag="realtime_detections"):
            pass

dpg.bind_font(global_font)
dpg.bind_theme(global_theme)

dpg.create_viewport(title='Battery Spotter Monitoring', width=1920, height=1080)

dpg.setup_dearpygui()
dpg.show_viewport()

# save_state(data)
# data = load_state()

mqtt_client_id = f"Battery Spotter Monitoring ({random.randint(0, 1000)})"
print(f"MQTT client_id: {mqtt_client_id}")

mqtt_client = mqtt.Client(mqtt_client_id)
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(mqtt_broker[0], mqtt_broker[1])
mqtt_client.subscribe("DETECTION")
mqtt_client.subscribe("IMAGE")
mqtt_client.loop_start()

time_delta = datetime.timedelta(days=1)
lock = threading.Lock()

cur_time = time.time()
last_time = cur_time

while dpg.is_dearpygui_running():
    cur_time = time.time()

    if cur_time - last_time > 0.1:
        cur_datetime = datetime.datetime.now()

        with lock:
            if dashboard:
                update_series(data, time_delta=time_delta, cur_datetime=cur_datetime)
                update_activity(data, time_delta=time_delta, cur_datetime=cur_datetime)
                update_clients(data, time_delta=time_delta, cur_datetime=cur_datetime)
                update_summary(data, time_delta=time_delta, cur_datetime=cur_datetime)
                update_detections(data, time_delta=time_delta, cur_datetime=cur_datetime)
            else:
                update_realtime_detections(data, client_name, cur_datetime)
                update_realtime_detections_history(data, client_name, cur_datetime)

        last_time = cur_time

    dpg.render_dearpygui_frame()

dpg.destroy_context()
