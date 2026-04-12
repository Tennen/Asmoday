# ROI 占用检测机制（PRD + Technical Design）

## 一、背景

现有系统主要依赖 YOLO 对指定区域做实体识别，再通过 zone + dwell 语义判断停留。

这条链路在以下场景里不够稳定：

- 实体与背景对比度低，例如白色目标在浅色地面。
- 实体进入后长时间静止，YOLO 置信度下降或漏检。
- 业务真正关心的是“区域是否仍被占用”，而不是每一帧都要重新识别“这是什么”。

因此需要引入 ROI 占用检测机制，并让它成为未来的常驻轻量检测方式，YOLO 退居为语义识别辅助链路。

## 二、目标

ROI 机制用于：

- 判断指定区域是否出现新增占用。
- 在实体静止或 YOLO 不稳定时，继续维持“区域被占用”的判断。
- 提供区域级状态：`empty` / `candidate_occupied` / `occupied`。
- 在 `roi_triggered` 模式下，作为 YOLO 的触发条件，降低持续算力开销。

ROI 不负责：

- 判断实体类别。
- 替代 YOLO 输出精确框、轨迹或分类。
- 在摄像头抖动、强反光、频繁全局光照变化下保证稳定。

## 三、核心结论

这是当前最实用、最容易落地的一版方案：

- 用 OpenCV `BackgroundSubtractorMOG2` 做 ROI 前景提取。
- 用 ROI mask 限定检测区域。
- 对前景 mask 做开运算 + 闭运算去噪。
- 计算两个指标：
  - `occupancy_ratio = roi_foreground_pixels / roi_area_pixels`
  - `largest_blob_area = 最大连通域面积`
- 单帧命中条件：
  - `occupancy_ratio >= 0.08`
  - 或 `largest_blob_area >= max(400, roi_area_pixels * 0.03)`
- 连续命中超过 `rule.stay_threshold_seconds` 判 `occupied`。
- 连续不命中超过 `clear_hold_seconds` 判 `unoccupied`。

其中 `clear_hold_seconds` 建议保留，默认 `1.0s`，不建议只要一帧不满足就立即判 `unoccupied`。

原因很直接：

- MOG2 对压缩噪声、局部反光、短时亮度变化会有抖动。
- 目标静止后，前景 mask 可能出现短暂收缩或断裂。
- 如果退出没有 debounce，状态会频繁抖动，后续 YOLO 启停和事件上报都会变得很差。

## 四、为什么选 MOG2

选择 `MOG2` 而不是简单帧差，原因是它更适合固定机位下的持续背景建模：

- 它是 OpenCV 官方提供、成熟且 CPU 成本可控的方案。
- 对慢变化背景比纯帧差稳。
- 能直接输出前景 mask，便于后续形态学和连通域分析。
- 参数数量适中，适合先做 v1，再根据现场数据调参。

调研时需要特别注意一个关键点：

- OpenCV 官方文档明确说明，若某个前景像素在大约 `backgroundRatio * history` 帧里保持近似稳定，它会被重新吸收到背景模型里。

这意味着：

- 单纯“跑 MOG2 然后看当前 mask”并不能天然解决“目标进入后长期静止”的问题。
- 如果不控制学习率，常驻目标最终仍会从 mask 中消失。

所以 v1 设计里必须加入动态学习率策略，而不是只靠默认参数硬扛。

## 五、推荐的 v1 架构

### 5.1 推荐实现粒度

v1 推荐按 `rule` 维护独立的 `ROIOccupancyDetector`，每个 detector 只处理自己的 ROI crop，而不是先做整帧共享前景再分发。

优点：

- 和当前 repo 的 rule-centric runtime 更一致，接入成本低。
- 每个 rule 的背景模型独立，某个 ROI 进入占用态时可以冻结自己的学习率，不会影响同摄像头其他区域。
- ROI crop 面积通常远小于整帧，CPU 开销仍然可控。

如果后续发现同一摄像头 ROI 数量很多、MOG2 成本明显上升，再做二阶段优化，演进到“摄像头级共享 subtractor + rule 级 mask/state machine”。

### 5.2 检测器职责

`ROIOccupancyDetector` 只负责：

- 维护本 rule 的背景模型。
- 维护本 rule 的 ROI mask。
- 输出当前单帧指标和稳定状态机结果。

YOLO 链路负责：

- 在 `always` 模式下持续提供实体语义。
- 在 `roi_triggered` 模式下，被 ROI 触发后启动，用于识别“是什么实体”。

## 六、OpenCV 逐帧实现方案

### 6.1 初始化

每个 rule 初始化以下对象：

- `cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=False)`
- 本地 ROI crop 坐标
- 本地 ROI mask
- 开运算 kernel：`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))`
- 闭运算 kernel：`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))`
- 状态机计时器：
  - `occupied_streak_seconds`
  - `clear_streak_seconds`
  - `warmup_remaining_seconds`

推荐默认值：

- `history = 300`
- `varThreshold = 16`
- `warmup_seconds = 3`
- `clear_hold_seconds = 1.0`

说明：

- `detectShadows` v1 默认关闭。阴影检测会把阴影像素标成单独的灰度值，虽然更细，但会让判定链路复杂化。v1 目标是先得到稳定的二值占用判断。
- `warmup_seconds` 期间只学习背景，不发占用事件。

### 6.2 ROI 输入

每帧对 rule 执行：

1. 按 `zone` 从原始 frame 裁出 ROI crop。
2. 如果 ROI crop 太大，按长边上限做等比缩放，建议 `max_side_px = 320`。
3. 生成或复用本地二值 ROI mask。

裁剪后再做 MOG2，而不是整帧先做 MOG2 再相交，原因是：

- v1 更简单。
- 背景模型只学习和该 rule 相关的像素。
- `largest_blob_area` 的阈值更容易和本 ROI 的工作分辨率绑定。

### 6.3 动态学习率

这是 v1 的关键设计。

状态不同，`MOG2.apply` 的学习率不同：

- `empty` 状态：`learningRate = -1`，使用 OpenCV 自动学习率。
- `candidate_occupied` 状态：`learningRate = 0`
- `occupied` 状态：`learningRate = 0`

这样做的原因是：

- 在 `empty` 状态，需要背景模型继续适应慢变化环境。
- 一旦区域开始出现持续前景，就不应该再让模型快速把该目标学成背景。
- 如果继续学习，静止目标迟早会被“吞掉”，这和 ROI 常驻占用检测的目标相反。

对于误触发导致的短暂 `candidate_occupied`：

- 只要候选状态在进入 `occupied` 前消失，就立即回到 `empty`。
- 回到 `empty` 后恢复背景学习。

### 6.4 每帧处理流程

最实用的一版逐帧流程如下：

1. `fgmask = subtractor.apply(roi_crop, learningRate=...)`
2. 若后续开启阴影检测，则先把 `fgmask == 255` 作为真实前景；v1 因为 `detectShadows=False`，直接把非零转成二值 mask。
3. `masked = cv2.bitwise_and(fgmask_binary, roi_mask)`
4. `opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, open_kernel)`
5. `cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)`
6. 计算：
   - `roi_foreground_pixels = cv2.countNonZero(cleaned)`
   - `roi_area_pixels = cv2.countNonZero(roi_mask)`
   - `occupancy_ratio = roi_foreground_pixels / roi_area_pixels`
7. 做连通域分析：
   - `num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)`
   - `largest_blob_area = max(stats[1:, cv2.CC_STAT_AREA])`，若没有前景则为 `0`
8. 单帧判定：
   - `frame_present = occupancy_ratio >= 0.08 or largest_blob_area >= max(400, roi_area_pixels * 0.03)`
9. 把 `frame_present` 送入状态机。

## 七、状态机定义

### 7.1 状态

- `warming_up`
- `empty`
- `candidate_occupied`
- `occupied`

### 7.2 进入 occupied

规则：

- 当 `frame_present=True` 连续累计达到 `rule.stay_threshold_seconds`，状态切换到 `occupied`。
- 这里直接复用现有 rule 的停留阈值，不额外新造“进入占用阈值”。

### 7.3 退出 occupied

规则：

- 当 `frame_present=False` 连续累计达到 `clear_hold_seconds`，状态切换到 `empty`。

推荐：

- 默认 `clear_hold_seconds = 1.0`
- 可通过环境变量配置，建议变量名：`VISION_SERVICE_ROI_CLEAR_HOLD_SECONDS`

不建议“只要不满足就立刻 unoccupied”，因为这样会带来：

- 状态抖动
- YOLO 频繁启停
- 事件 spam
- 证据截图不稳定

### 7.4 候选状态的意义

`candidate_occupied` 不是多余状态，它有两个作用：

- 提前冻结学习率，避免静止目标在进入 `occupied` 前被吸收进背景。
- 在 `roi_triggered` 模式下，可以更早启动 YOLO，不必等到完整 dwell 阈值已经满足。

## 八、阈值建议

### 8.1 `occupancy_ratio`

推荐默认值：`0.08`

解释：

- 对中等大小 ROI 来说，8% 的前景面积已经足够说明“区域里确实进来了东西”，又不会因为少量边缘噪声就频繁命中。
- 这是一个适合先上线采样数据的起始值，不应被视为永远固定。

### 8.2 `largest_blob_area`

不建议直接写死成一个全局像素值，因为 ROI 会缩放、摄像头分辨率也不一致。

v1 推荐公式：

- `largest_blob_area_threshold = max(400, roi_area_pixels * 0.03)`

理由：

- 既保留“最大连通域不能太小”的约束，
- 又避免不同分辨率下阈值完全失真。

### 8.3 为什么同时保留两个条件

仅看 `occupancy_ratio` 不够：

- 当前景被切碎成几块时，面积可能够，但其实只是噪声。

仅看 `largest_blob_area` 也不够：

- 一些真实占用可能是分散前景，单个 blob 不大，但总体占比已经明显上升。

因此 v1 保留双条件 `OR` 更稳。

## 九、与 YOLO 的配合方式

### 9.1 `always`

- ROI 常驻运行。
- YOLO 也常驻运行。
- 当 YOLO 漏检或不稳定时，ROI 负责把“仍然 occupied”的状态维持住。

### 9.2 `roi_triggered`

- ROI 常驻运行。
- 当某个 rule 进入 `candidate_occupied` 时启动 YOLO。
- 当该 rule 重新回到 `empty` 且超过 `clear_hold_seconds` 后停止 YOLO。

不建议等到 `occupied` 才启动 YOLO，因为那样会错过刚进入区域时最容易识别实体类别的时间窗口。

## 十、与当前 Gateway contract 的关系

需要明确区分两层语义：

- ROI 内部状态机语义：`empty` / `candidate_occupied` / `occupied`
- Gateway 对外事件语义：当前 contract 仍是 `threshold_met` / `cleared`

因此：

- `occupied/unoccupied` 首先应视为内部执行状态，不应直接当成 wire status 发出去。
- 后续实现时，需要由上层 runtime 把 ROI 内部状态映射到现有事件协议，或者先把 ROI 仅作为 YOLO 启停和占用维持信号。
- 在没有修改 `vision-service-contract.md` 之前，不应擅自把外部事件名改成 `occupied/unoccupied`。

## 十一、限制与边界

### 11.1 启动阶段

如果服务启动时 ROI 内已经有一个长期静止目标：

- MOG2 可能会在 warmup 期间把它直接当成背景。
- 因此 v1 更擅长检测“新增进入并停留”，不擅长在服务启动瞬间盘点“已经存在的静止占用”。

这个限制需要在文档中明确，不要假装没有。

### 11.2 不适用场景

- 摄像头机位变化
- 剧烈全局光照变化
- 大面积反光、水波纹、电视屏幕等动态背景
- 业务必须严格区分实体类型

### 11.3 这不是 YOLO 替代品

ROI 提供的是“区域被占住了”的低成本状态信号，不提供稳定类别语义。

## 十二、建议的代码拆分

为了符合当前仓库的文件拆分约束，后续实现建议拆成：

- `src/vision_service/vision/roi/detector.py`
- `src/vision_service/vision/roi/state.py`
- `src/vision_service/vision/roi/models.py`
- `tests/vision/roi/test_detector.py`
- `tests/vision/roi/test_state.py`

避免把 ROI 检测、状态机、YOLO 联动逻辑继续堆进现有单个大文件。

## 十三、伪代码

```python
def process_frame(frame: np.ndarray, observed_at: datetime) -> ROIFrameResult:
    roi_crop = crop_and_resize(frame, zone, max_side_px=320)
    learning_rate = choose_learning_rate(state)
    fgmask = subtractor.apply(roi_crop, learningRate=learning_rate)

    fgmask_binary = (fgmask > 0).astype("uint8") * 255
    masked = cv2.bitwise_and(fgmask_binary, roi_mask)
    opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)

    roi_pixels = cv2.countNonZero(roi_mask)
    foreground_pixels = cv2.countNonZero(cleaned)
    occupancy_ratio = foreground_pixels / roi_pixels if roi_pixels else 0.0

    _, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    largest_blob_area = (
        int(stats[1:, cv2.CC_STAT_AREA].max())
        if len(stats) > 1
        else 0
    )

    frame_present = (
        occupancy_ratio >= 0.08
        or largest_blob_area >= max(400, int(roi_pixels * 0.03))
    )

    return state_machine.observe(
        observed_at=observed_at,
        frame_present=frame_present,
        occupancy_ratio=occupancy_ratio,
        largest_blob_area=largest_blob_area,
    )
```

## 十四、后续实现时必须补的测试

- ROI 内无前景时保持 `empty`
- ROI 内连续前景达到 `stay_threshold_seconds` 后进入 `occupied`
- 占用后短暂 1-2 帧丢失前景，不应立即 `unoccupied`
- 持续空白超过 `clear_hold_seconds` 后退出 `occupied`
- 静止目标进入后，因学习率冻结而不会快速被背景模型吞掉
- 启动时 ROI 已被占用的限制行为要有明确测试和文档

## 十五、调研参考

- OpenCV `BackgroundSubtractorMOG2` 官方文档：
  [https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
- OpenCV 形态学操作教程：
  [https://docs.opencv.org/master/d3/dbe/tutorial_opening_closing_hats.html](https://docs.opencv.org/master/d3/dbe/tutorial_opening_closing_hats.html)
- OpenCV `connectedComponentsWithStats` / `CC_STAT_AREA` 文档：
  [https://docs.opencv.org/4.x/javadoc/org/opencv/imgproc/Imgproc.html](https://docs.opencv.org/4.x/javadoc/org/opencv/imgproc/Imgproc.html)
