# # import time
# # while infinite_loop:
# #     if keystroke:
# #         stop_flag = True
# #     if stop_flag:
# #         import ipdb;ipdb.set_trace()
# #         stop_flag = False
# #     time.sleep(2)

# import pynput

# flag = False

# def on_press(key):
#     global flag
#     if key == pynput.keyboard.Key.ctrl_l and pynput.keyboard.KeyCode(char='j'):
#         flag = True
#     print('onpressed')

# with pynput.keyboard.Listener(on_press=on_press) as listener:
#     listener.join()
# print('hello?')

import evdev

flag = False

device = evdev.InputDevice('/dev/input/event0')

for event in device.read_loop():
    if event.type == evdev.ecodes.EV_KEY:
        data = evdev.categorize(event)
        if data.keystate == 1: # Key press
            if data.keycode == 'KEY_LEFTCTRL' and data.keycode == 'KEY_J':
                flag = True
                break
