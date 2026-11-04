# 最新范围：其余四类各50题（2026-11-02）

用户明确选“六类各50题”，随后明确“之前已经有的两类就不跑”。最终新生成范围因此只有易混淆视角、视觉谜题、复用图像、自拍图像，分别取作者原顺序前50条，共200道问答，每题中性/原始正确提示/原始错误提示三条件。OCR、伪装不再增加任何新生成；已有结果保留，不擅自补齐类别或强行凑成所有模型各300题。

所有模型使用同一200题。BF16/SDPA，原模型各自图像与采样配置、seed42、4096思考+512回答保持；只做原模型baseline，无ITSM/S2VA/注意力干预。PENDULUM语义标签pending，遵守work/annotation_policy.md，不把格式异常直接判错。

## 已部署

- 59的GPU0、1：Qwen3.8两分片，每卡处理90题×3条件=270条。原source_row<5的20题由51的健康pilot30继续完成，禁止在59重新生成。有限CPU收集步骤collect_existing_qwen_prefix.py等待这60份已有分配的输出后，逐份SHA原样汇入59；会保存prefix_import_receipt.json。总目标600份。
- 59的GPU2、3：Qwen3.6-35B-A3B MoE两分片，每卡100题×3条件=300条，共600份。使用59原有约67GiB BF16权重，未传输任何权重。59新快照65项pytest全部通过。
- 51的GPU3、5：原Gemma30题先自然完成；CPU driver103072之后复用其中与这200题重合的60份输出，只补540份。新worker固定检查四类各50，完成600份即退出。driver不占GPU、不停止pilot、不自动重试失败worker。
- 51的GPU1、2当前仅完成原Qwen30题，结束即释放，没有接回旧全量。
- 51 GPU4完全不动。59 GPU4–7有他人的进程，不动。

两台远端根均为/mnt/sdb/linxin/pendulum-20261102/expand4x50。worker.py按model/shard写job_state_k.json、run_k.log、worker_k.exit及records。原大规模记录仍留在父目录，未删除、未重启。

## 来源和运行版本

固定200题原顺序、图片指纹和提示保留，数据共84张去重图片、约23MB；已复制到59并逐张SHA验证。59使用51已验证的同一隔离源码快照，未改正在跑的仓库。

51 torch2.8.0+cu128、transformers5.20.2；59 torch2.11.0+cu130、transformers5.14.1。保留各自原有环境，不安装或修改环境。Qwen3.8首20题来自51、其余180来自59；来源与版本在状态、配置和复用清单中保留，不能冒称同设备同软件版本的严格复现。

状态读取status_expansion.py。当前最后检查：59四个worker均generating且PID存在，Qwen已保存32份，MoE已保存5份；51Gemma接续正在等待既有pilot。此为快照，不是持续实时状态。所有模型生成保存完成自然退出释放GPU；没有新heartbeat或占卡进程。

如SSH连接断开，有限收集步骤可能失败并留prefix_collection_state.json；保留文件，不声称60条已导入，也不要重新生成那20题。

用户随后撤销额外GPU加速：两个新增CPU等待调度器在模型启动前已取消，GPU6未用于新模型。原Gemma GPU3、5接续安排已恢复，新CPU scheduler PID=329644。没有中断任何健康生成。


2026-11-02：新增用户授权的空间注意力收集，固定re_used/self_captured各50题，三条件，每模型300份只读回放。详细见attention_collection.md。51 Gemma GPU1,2已启动并首份原前向逐位一致；59 Qwen/MoE有限队列等待各自生成worker0成功退出后用GPU0/2接续。原生成进程与参数不变，GPU4不动，GPU6不使用。


2026-11-02晚，用户重新明确授权加速：59空闲GPU0、1用于现有MoE；51优先生成，暂停占GPU等待源回复的注意力收集。已部署parallel_acceleration/下的有限交接，MoE由2生成进程改4（59 GPU2/3/0/1），Gemma由1改2（51 GPU3,5和1,2）。新增模型先预载，原生成在下一份回复保存后TERM并接共享互斥队列；输入、模型、BF16、seed42、采样、预算均保持。已有回复按SHA保留，无重复生成。51原30份注意力保留，生成后按源/数组SHA验证后接续剩余270份。59原注意力有限队列保留。51 GPU6单张放不下同一Gemma，不建立占卡进程。当前实际交接状态以parallel_acceleration/latest_59.json和latest_51.json及远端handoff_state.json为准，不把部署当作已开始生成。
