# 在任意机器上给 USB 脚踏板接入 Python 脚本

本文档说明如何把一对 USB HID 脚踏板（型号 **FS-01**, 芯片 **LinTx VID=8088 PID=0015**）接入任何基于 `lerobot.utils.control_utils.init_keyboard_listener` 的录制脚本，或独立的 Python 脚本。

核心设计：**直接读 `/dev/input/event*`**，绕过 TTY / X / Wayland / 窗口焦点，**任何进程——包括 SSH headless、tmux、systemd service——只要能打开事件节点就能收到按键**。

---

## 1. 硬件识别

先确认手上这对脚踏板是不是 LinTx 芯片方案（同一 model name "FS-01" 有多家代工）。插上后跑：

```bash
lsusb | grep -i '8088:0015'
cat /proc/bus/input/devices | grep -B1 -A6 -i lintx
```

应当看到类似：

```
Bus 001 Device 022: ID 8088:0015
Bus 001 Device 021: ID 8088:0015

I: Bus=0003 Vendor=8088 Product=0015 Version=0200
N: Name="LinTx LinTx Keyboard"
P: Phys=usb-0000:00:14.0-5.2/input1
S: Sysfs=/devices/.../input/input20
U: Uniq=BE1072C8
H: Handlers=sysrq kbd event19 leds
```

两只踏板 **VID/PID 相同**（都是 `8088:0015`），但 LinTx 固件里每只都有独立序列号写在 `U: Uniq=...` 字段——这是区分它们的唯一可靠方式。

记录下两个关键信息：

1. **每只踏板的 serial**（`Uniq=` 后面的值，例如 `BE1072C8` 和 `BE136B2F`）
2. **by-id 稳定符号链接**（换 USB 口不失效）：
   ```bash
   ls -la /dev/input/by-id/usb-LinTx_LinTx_Keyboard_*-if01-event-kbd
   ```
   输出会是：
   ```
   /dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE1072C8-if01-event-kbd -> ../event19
   /dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE136B2F-if01-event-kbd -> ../event21
   ```

> **如果 VID/PID 不是 `8088:0015`**：说明你这对踏板是另一家代工（比如真正的 PCsensor `0c45:7403`），整套 udev 规则仍然可用，只是把规则里的 `idVendor`/`idProduct` 换成实际值。如果你想做真正的固件写入（而不是 udev 软重映射），PCsensor 方案可以用 [rgerganov/footswitch](https://github.com/rgerganov/footswitch)；LinTx 方案没有公开的烧录工具，只能走 udev 软重映射。

---

## 2. 默认行为：两只都输出空格

LinTx FS-01 出厂固件里两只踏板都是 HID keyboard，按下发 `KEY_SPACE`（HID usage page 0x07, usage 0x2c）。直接接 Python 读事件节点会两只都看到 `KEY_SPACE`，没法区分。解决方法有两个：

- **方法 A（推荐，零代码开销）**：用 udev `KEYBOARD_KEY_*` 机制把其中一只软重映射成别的键（例如 `r`），`EVIOCSKEYCODE` ioctl 由内核处理，Python 读事件时直接看到 `KEY_R`。
- **方法 B**：不做重映射，Python 按 device path（serial）区分两只踏板，内部维护"哪只 path 意味着什么语义"的映射。

本项目采用 **方法 A**，因为它让上层 Python 代码完全无感知——一只踏板 = 一种按键，跟真实键盘体验一致。

### HID scancode 速查

udev `KEYBOARD_KEY_*` 机制需要 HID 的 raw scancode（`MSC_SCAN` 值）。对于 HID keyboard（usage page 0x07），scancode 的格式是 `0x7`（page）拼 usage：

| 按键 | HID usage | scancode (hex) |
|---|---|---|
| a | 0x04 | `70004` |
| r | 0x15 | `70015` |
| s | 0x16 | `70016` |
| f | 0x09 | `70009` |
| space | 0x2c | `7002c` |
| enter | 0x28 | `70028` |
| esc | 0x29 | `70029` |
| tab | 0x2b | `7002b` |

如果不确定真实值，用 `evtest` 读一下：

```bash
sudo apt-get install -y evtest
sudo evtest /dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE1072C8-if01-event-kbd
# 踩一下，找形如：
#   Event: type 4 (EV_MSC), code 4 (MSC_SCAN), value 7002c
```

---

## 3. 安装 udev 规则：权限 + 可选重映射

写一个 udev 规则同时做两件事：
1. 给两只 LinTx 设备授予 `plugdev:0660` 权限（这样普通用户不用 sudo 能 `open()` 事件节点）
2. 把其中一只踏板的 space 重映射成另一个键

```bash
sudo tee /etc/udev/rules.d/90-lintx-pedal-remap.rules > /dev/null <<'EOF'
# LinTx foot pedal: remap Pedal A SPACE->R + grant plugdev read on both
ACTION=="remove", GOTO="lintx_pedal_end"
SUBSYSTEM!="input", GOTO="lintx_pedal_end"
KERNEL!="event*", GOTO="lintx_pedal_end"

# Permissions: both LinTx pedals readable by plugdev group
ATTRS{idVendor}=="8088", ATTRS{idProduct}=="0015", GROUP="plugdev", MODE="0660"

# Remap Pedal A (serial BE1072C8): HID scancode 7002c (space) -> r
ATTRS{idVendor}=="8088", ATTRS{idProduct}=="0015", ATTRS{serial}=="BE1072C8", ENV{KEYBOARD_KEY_7002c}="r", RUN{builtin}+="keyboard"

LABEL="lintx_pedal_end"
EOF

sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=input --action=change
```

> **换机器时要改的参数**：
> 1. `ATTRS{serial}` 换成你那台机器上想被重映射的踏板的 serial（从 Step 1 的 `Uniq=` 读到）
> 2. 如果芯片不是 LinTx，`idVendor`/`idProduct` 换成 `lsusb` 输出里看到的值
> 3. 如果要重映射成 `r` 以外的键，`KEYBOARD_KEY_` 后面的 scancode 参考 Step 2 的速查表，等号右边的值是内核 key name 小写（`r`、`enter`、`space`、`esc`、...）

### 验证

```bash
# 事件节点权限应该是 root:plugdev 0660
ls -la /dev/input/event19 /dev/input/event21

# udevadm test 应该显示 KEYBOARD_KEY_7002c=r 被应用
udevadm test /sys/class/input/event19 2>&1 | grep -iE '90-lintx|keyboard_key'

# evtest 实际踩一下，重映射过的那只应该发 KEY_R
sudo evtest /dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE1072C8-if01-event-kbd
```

### 用户必须在 plugdev 组

```bash
groups | grep -o plugdev || sudo usermod -aG plugdev $USER
# 注意：usermod 后需要重新登录（或用 newgrp plugdev）才生效
```

---

## 4. Python 集成

### 4.1 依赖

```bash
pip install evdev
```

### 4.2 方式 A：独立的 `PedalListener` 回调

适用于任何 Python 脚本（不限 lerobot），后台 daemon 线程 + `select`，非阻塞：

```python
from lerobot.utils.pedal_listener import PedalListener

def on_pedal(key: str) -> None:
    if key == 'r':
        print("Pedal A pressed — start RL phase")
    elif key == 'space':
        print("Pedal B pressed — toggle intervention")

listener = PedalListener(on_pedal)
if not listener.start():
    # 设备不存在或不可读 —— 静默降级，不报错
    print("no pedal detected; continuing without")

# ... 你的主循环
# listener.stop()  # 可选；daemon 线程会随进程退出
```

**自定义设备路径**（换机器时）：

```python
listener = PedalListener(
    on_press=on_pedal,
    devices=(
        "/dev/input/by-id/usb-LinTx_LinTx_Keyboard_<serial1>-if01-event-kbd",
        "/dev/input/by-id/usb-LinTx_LinTx_Keyboard_<serial2>-if01-event-kbd",
    ),
)
```

**自定义按键映射**（如果你没用 udev 重映射，或者用了别的 scancode）：

```python
from evdev import ecodes

listener = PedalListener(
    on_press=on_pedal,
    key_map={
        ecodes.KEY_R: 'r',
        ecodes.KEY_SPACE: 'space',
        ecodes.KEY_ENTER: 'enter',  # 比如第三只踏板发 ENTER
    },
)
```

### 4.3 方式 B：通过 `init_keyboard_listener` 自动集成

任何调用 `init_keyboard_listener(...)` 的 lerobot 脚本（`lerobot_record.py`、`record_rlt_hil.py` 等）**零改动自动支持脚踏板**——`init_keyboard_listener` 内部无条件启动一个 `PedalListener`，设备不存在就静默跳过。

映射规则是 data-driven 的，跟随你传给 `init_keyboard_listener` 的 key 参数：

```python
from lerobot.utils.control_utils import init_keyboard_listener

listener, events = init_keyboard_listener(
    intervention_toggle_key=" ",   # 踩 Pedal B (space) -> events["toggle_intervention"] = True
    rl_phase_key="r",              # 踩 Pedal A (r)     -> events["start_rl_phase"] = True
    cp_success_key="s",            # 这些如果你让某只踏板发它们，也会自动路由
    cp_failure_key="f",
)
```

路由表构建逻辑（见 `control_utils.py::_start_pedal_listener`）：
- 按 first-win 规则注册：`rl_phase_key` → `intervention_toggle_key` → `critical_phase_toggle_key` → `cp_success_key` → `cp_failure_key` → `end_success_key` → `end_failure_key` → `episode_success_key` → `episode_failure_key`
- 空格键会被标准化成 `'space'`，其他字符小写
- `toggle_intervention` 和 `toggle_critical_phase` 自带防抖（`INTERVENTION_TOGGLE_COOLDOWN_S = 0.5` 秒）

> 只要 udev 重映射后 Pedal A 发 `r`、Pedal B 发 `space`，这两只踏板就会自动按现有的键盘 `r` / 空格 的语义触发 `events` 字段，主循环不用做任何改动。

---

## 5. End-to-end smoke test

验证权限 + 重映射 + PedalListener 全链路：

```bash
# 1) PedalListener 纯捕获测试（30 秒窗口，踩两只踏板）
PYTHONPATH=src python -c "
import time
from lerobot.utils.pedal_listener import PedalListener

events = []
def on_press(k):
    events.append(k)
    print(f'>>> PEDAL {k.upper()} <<<', flush=True)

pl = PedalListener(on_press)
assert pl.start(), 'no pedal detected'
print('press both pedals within 30s...', flush=True)
time.sleep(30)
print(f'captured: {events}')
"

# 2) init_keyboard_listener 集成测试
PYTHONPATH=src python -c "
import time
from lerobot.utils.control_utils import init_keyboard_listener
listener, events = init_keyboard_listener(intervention_toggle_key=' ', rl_phase_key='r')
print('press pedals within 20s...', flush=True)
t_end = time.time() + 20
while time.time() < t_end:
    if events['start_rl_phase'] and events['toggle_intervention']:
        break
    time.sleep(0.1)
print('start_rl_phase =', events['start_rl_phase'])
print('toggle_intervention =', events['toggle_intervention'])
"
```

期望：两个 case 都应该看到踏板事件写入。

---

## 6. 常见坑

### 6.1 权限：`Permission denied` / `start() -> False`

```
Pedal device not readable (check udev rule for plugdev:0660):
    /dev/input/by-id/usb-LinTx_LinTx_Keyboard_BE1072C8-if01-event-kbd
```

**原因**：udev 规则没生效 或 用户不在 `plugdev` 组 或 `usermod` 后没重登录。

**排查**：

```bash
ls -la /dev/input/event19              # 必须显示 root:plugdev 0660
groups                                  # 必须包含 plugdev
udevadm info --query=property --name=/dev/input/event19 | grep -i group
sudo udevadm trigger --subsystem-match=input --action=change   # 重新跑 udev rule
```

### 6.2 重映射没生效：还是发 space

```
evtest .../usb-LinTx_LinTx_Keyboard_BE1072C8-if01-event-kbd
# type 1 (EV_KEY), code 57 (KEY_SPACE)  ← 应该是 code 19 (KEY_R)
```

**可能原因**：

1. **scancode 填错了**：HID 不一定是 `7002c`。用 `sudo evtest` 读真实的 `MSC_SCAN` 值：
   ```
   Event: type 4 (EV_MSC), code 4 (MSC_SCAN), value 7002c
   ```
   把规则里的 `KEYBOARD_KEY_7002c` 改成实际值。
2. **udev 规则文件写错了** `ATTRS{serial}`、`idVendor`、`idProduct`：用 `udevadm test /sys/class/input/event19` 看它匹配了哪些规则。
3. **内核版本太老**（< 4.x）不支持 `RUN{builtin}+="keyboard"`：极少遇到，Ubuntu 20.04+ 都 OK。
4. **规则 reload 没跑到**：`sudo udevadm control --reload && sudo udevadm trigger ...`。

### 6.3 `PedalListener` 启动成功但收不到事件

**可能原因**：

1. **另一个进程 grab 了设备独占**（比如另一个正在跑的 evtest `--grab`）。`fuser /dev/input/event19` 看谁在占。
2. **回调里抛异常了**：看日志里是否有 `PedalListener callback raised` 的 traceback。回调异常会被记录但不会杀掉 listener 线程。
3. **线程没启动**：检查 `pl.start()` 是否返回 `True`。

### 6.4 SSH 进来 `which python` 为空

**原因**：SSH 非登录 shell 没有源 `~/.bashrc`，conda 不会自动 activate。

**解决**：用 conda env 的绝对路径，或者在命令前 source：

```bash
# 绝对路径（推荐，脚本化更稳）
/home/<user>/miniconda3/envs/<env>/bin/python -c "..."

# 或者 source
source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env> && python -c "..."
```

### 6.5 两只踏板序列号是一样的（固件批次问题）

极少数情况下同批次出厂的 LinTx 会写入相同 `Uniq`。这时没法用 `ATTRS{serial}` 区分，只能退回用 **USB 端口路径**：

```bash
# /proc/bus/input/devices 里的 Phys 字段
P: Phys=usb-0000:00:14.0-5.2/input1
P: Phys=usb-0000:00:14.0-5.3/input1
```

udev 规则里用 `KERNELS=="1-5.2"` / `KERNELS=="1-5.3"` 匹配。缺点：换 USB 口后规则失效，每次插到不同口都要调整。

---

## 7. 文件索引

- `src/lerobot/utils/pedal_listener.py` — 通用 `PedalListener` 类
- `src/lerobot/utils/control_utils.py::_start_pedal_listener` — lerobot 集成层
- `src/lerobot/utils/control_utils.py::init_keyboard_listener` — 自动挂载点（无条件调用）
- `/etc/udev/rules.d/90-lintx-pedal-remap.rules` — 部署机器上的 udev 规则（不在 git 里，每台机器单独写）
