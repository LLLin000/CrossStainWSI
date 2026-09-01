"""
CrossStainWSI 图形界面工作台 (GUI Workbench)
基于标准库 Tkinter / ttk 构建，与底层计算引擎严格解耦
"""

from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from crossstainwsi.inventory.assets import AssetInventory, SampleAssets
from crossstainwsi.inventory.discover import AssetDiscoverer
from crossstainwsi.pipeline.config import PipelineConfig
from crossstainwsi.pipeline.sample_runner import SampleRunner
from crossstainwsi.planning.goal import StainRequirement, UserGoal, ViewSpec
from crossstainwsi.planning.planner import WorkflowPlanner
from crossstainwsi.review.states import RunVerdict


class CrossStainWSIGUI:
    """
    CrossStainWSI 独立 GUI 应用程序
    """
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CrossStainWSI - 多染色病理切片自动配准与提取工作台")
        self.root.geometry("1100x720")
        self.root.minsize(960, 600)

        # 状态数据
        self.inventory: Optional[AssetInventory] = None
        self.sample_rows_data: Dict[str, Dict] = {} # sample_id -> {ref_stain, mirror, status, plan, assets}
        self.is_running = False
        self.stop_requested = False

        # 设置样式
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._build_ui()

    def _build_ui(self):
        # 1. 主滚动/布局容器
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ================= Top: 目录与全局设置面板 =================
        top_group = ttk.LabelFrame(main_frame, text=" 1. 目录与全局配置 ", padding=10)
        top_group.pack(fill=tk.X, pady=(0, 10))

        # 路径网格
        ttk.Label(top_group, text="切片 WSI 目录:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.wsi_dir_var = tk.StringVar(value=r"E:\研究数据\骨科\切片扫描\2026-08-21")
        ttk.Entry(top_group, textvariable=self.wsi_dir_var, width=50).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=3)
        ttk.Button(top_group, text="浏览...", command=self._browse_wsi_dir).grid(row=0, column=2, padx=2, pady=3)

        ttk.Label(top_group, text="截图 Evidence 目录:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.tiff_dir_var = tk.StringVar(value=r"E:\研究数据\骨科\切片扫描\tiff")
        ttk.Entry(top_group, textvariable=self.tiff_dir_var, width=50).grid(row=1, column=1, sticky=tk.EW, padx=5, pady=3)
        ttk.Button(top_group, text="浏览...", command=self._browse_tiff_dir).grid(row=1, column=2, padx=2, pady=3)

        ttk.Label(top_group, text="结果输出目录:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.out_dir_var = tk.StringVar(value=r"E:\研究数据\骨科\切片扫描\registered_crops_300dpi")
        ttk.Entry(top_group, textvariable=self.out_dir_var, width=50).grid(row=2, column=1, sticky=tk.EW, padx=5, pady=3)
        ttk.Button(top_group, text="浏览...", command=self._browse_out_dir).grid(row=2, column=2, padx=2, pady=3)

        top_group.columnconfigure(1, weight=1)

        # 参数行
        param_frame = ttk.Frame(top_group)
        param_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))

        ttk.Label(param_frame, text="默认基准染色:").pack(side=tk.LEFT, padx=(0, 4))
        self.ref_stain_var = tk.StringVar(value="masson")
        ttk.Combobox(param_frame, textvariable=self.ref_stain_var, values=["masson", "HE", "Gram", "IHC"], width=8).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(param_frame, text="输出 DPI:").pack(side=tk.LEFT, padx=(0, 4))
        self.dpi_var = tk.StringVar(value="300")
        ttk.Combobox(param_frame, textvariable=self.dpi_var, values=["300", "600", "150"], width=6).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(param_frame, text="输出倍率:").pack(side=tk.LEFT, padx=(0, 4))
        self.mag_4x_var = tk.BooleanVar(value=True)
        self.mag_10x_var = tk.BooleanVar(value=False)
        self.mag_20x_var = tk.BooleanVar(value=True)
        self.mag_40x_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(param_frame, text="4×", variable=self.mag_4x_var).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(param_frame, text="10×", variable=self.mag_10x_var).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(param_frame, text="20×", variable=self.mag_20x_var).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(param_frame, text="40×", variable=self.mag_40x_var).pack(side=tk.LEFT, padx=3)

        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="生成 Overlay", variable=self.overlay_var).pack(side=tk.LEFT, padx=(15, 15))

        ttk.Button(param_frame, text="🔍 扫描并生成批次清单", command=self._scan_assets).pack(side=tk.RIGHT, padx=5)

        # ================= Middle: 样本列表与参数微调 =================
        mid_group = ttk.LabelFrame(main_frame, text=" 2. 待处理样本清单 (双击行可切换水平镜像纠偏) ", padding=10)
        mid_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ("sample_id", "ref_stain", "target_stains", "evidence", "mirror", "tier", "status")
        self.tree = ttk.Treeview(mid_group, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("sample_id", text="样本号")
        self.tree.heading("ref_stain", text="基准染色")
        self.tree.heading("target_stains", text="已发现切片")
        self.tree.heading("evidence", text="输入证据 (截图)")
        self.tree.heading("mirror", text="水平镜像纠偏?")
        self.tree.heading("tier", text="置信度等级")
        self.tree.heading("status", text="执行状态")

        self.tree.column("sample_id", width=120, anchor=tk.CENTER)
        self.tree.column("ref_stain", width=90, anchor=tk.CENTER)
        self.tree.column("target_stains", width=180, anchor=tk.W)
        self.tree.column("evidence", width=150, anchor=tk.W)
        self.tree.column("mirror", width=110, anchor=tk.CENTER)
        self.tree.column("tier", width=140, anchor=tk.CENTER)
        self.tree.column("status", width=100, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(mid_group, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self._toggle_row_mirror)

        # ================= Bottom: 进度与操作栏 =================
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))

        info_bar = ttk.Frame(bottom_frame)
        info_bar.pack(fill=tk.X, pady=(0, 6))

        self.status_label_var = tk.StringVar(value="准备就绪。点击上方【扫描并生成批次清单】开始。")
        ttk.Label(info_bar, textvariable=self.status_label_var, foreground="#333333").pack(side=tk.LEFT)

        btn_bar = ttk.Frame(bottom_frame)
        btn_bar.pack(fill=tk.X)

        self.start_btn = ttk.Button(btn_bar, text="▶ 开始批量配准提取", command=self._start_batch)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.stop_btn = ttk.Button(btn_bar, text="⏹ 停止", command=self._stop_batch, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(btn_bar, text="📋 查看选中计划", command=self._view_selected_plan).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_bar, text="📂 打开输出目录", command=self._open_out_dir).pack(side=tk.RIGHT)

    def _browse_wsi_dir(self):
        d = filedialog.askdirectory(initialdir=self.wsi_dir_var.get())
        if d:
            self.wsi_dir_var.set(d)

    def _browse_tiff_dir(self):
        d = filedialog.askdirectory(initialdir=self.tiff_dir_var.get())
        if d:
            self.tiff_dir_var.set(d)

    def _browse_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir_var.get())
        if d:
            self.out_dir_var.set(d)

    def _get_requested_mags(self) -> List[str]:
        mags = []
        if self.mag_4x_var.get():
            mags.append("4x")
        if self.mag_10x_var.get():
            mags.append("10x")
        if self.mag_20x_var.get():
            mags.append("20x")
        if self.mag_40x_var.get():
            mags.append("40x")
        return mags or ["4x", "20x"]

    def _scan_assets(self):
        base_dir = Path(self.wsi_dir_var.get())
        tiff_dir = Path(self.tiff_dir_var.get())

        if not base_dir.exists():
            messagebox.showerror("路径错误", f"切片 WSI 目录不存在:\n{base_dir}")
            return

        self.tree.delete(*self.tree.get_children())
        self.sample_rows_data.clear()

        # 执行资产扫描
        discoverer = AssetDiscoverer(base_dir=base_dir, tiff_dir=tiff_dir)
        self.inventory = discoverer.discover()

        # 默认镜像列表 (如 2-2w-1, 5-4w-2)
        default_mirrored = {"2-2w-1", "5-4w-2"}

        goal = UserGoal.from_magnifications(
            mags=self._get_requested_mags(),
            reference_stain=self.ref_stain_var.get(),
            dpi=int(self.dpi_var.get()),
        )
        planner = WorkflowPlanner(goal=goal)

        for sid, sample_asset in sorted(self.inventory.samples.items()):
            is_mirror = sid.lower() in [m.lower() for m in default_mirrored]
            sample_asset.roi_evidence.is_mirrored = is_mirror
            plan = planner.plan(sample_asset)

            slides_str = ", ".join(sample_asset.slides.keys())
            ev_str = "无"
            if sample_asset.roi_evidence.has_4x and sample_asset.roi_evidence.has_20x:
                ev_str = "4x + 20x"
            elif sample_asset.roi_evidence.has_4x:
                ev_str = "4x 截图"
            elif sample_asset.roi_evidence.has_20x:
                ev_str = "20x 截图"

            mirror_str = "✔ 是 (翻转)" if is_mirror else "否"

            item_id = self.tree.insert(
                "",
                tk.END,
                values=(
                    sid,
                    plan.reference_stain,
                    slides_str,
                    ev_str,
                    mirror_str,
                    plan.confidence_tier.value,
                    "就绪" if plan.is_executable else "缺少必选切片",
                ),
            )

            self.sample_rows_data[sid] = {
                "tree_item_id": item_id,
                "sample_id": sid,
                "assets": sample_asset,
                "plan": plan,
                "is_mirror": is_mirror,
                "status": "就绪",
            }

        self.status_label_var.set(f"扫描完成：发现 {len(self.sample_rows_data)} 个样本。准备就绪。")

    def _toggle_row_mirror(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        item_id = selected[0]
        values = list(self.tree.item(item_id, "values"))
        sid = values[0]
        if sid in self.sample_rows_data:
            current = self.sample_rows_data[sid]["is_mirror"]
            new_val = not current
            self.sample_rows_data[sid]["is_mirror"] = new_val
            values[4] = "✔ 是 (翻转)" if new_val else "否"
            self.tree.item(item_id, values=values)

    def _view_selected_plan(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在列表中选中一个样本。")
            return
        item_id = selected[0]
        sid = self.tree.item(item_id, "values")[0]
        row_info = self.sample_rows_data.get(sid)
        if row_info and row_info["plan"]:
            desc = row_info["plan"].describe()
            messagebox.showinfo(f"样本 {sid} 执行计划", desc)

    def _open_out_dir(self):
        import os
        d = self.out_dir_var.get()
        if Path(d).exists():
            os.startfile(d)
        else:
            messagebox.showwarning("提示", f"输出目录尚未创建:\n{d}")

    def _start_batch(self):
        if not self.sample_rows_data:
            messagebox.showinfo("提示", "当前无就绪样本，请先点击扫描。")
            return
        if self.is_running:
            return

        self.is_running = True
        self.stop_requested = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        # 启动后台工作线程，绝对不阻塞 UI
        threading.Thread(target=self._worker_thread, daemon=True).start()

    def _stop_batch(self):
        if self.is_running:
            self.stop_requested = True
            self.status_label_var.set("正在安全中止...")

    def _worker_thread(self):
        dpi_val = int(self.dpi_var.get())
        goal = UserGoal.from_magnifications(
            mags=self._get_requested_mags(),
            reference_stain=self.ref_stain_var.get(),
            dpi=dpi_val,
        )

        cfg = PipelineConfig(
            base_dir=Path(self.wsi_dir_var.get()),
            tiff_dir=Path(self.tiff_dir_var.get()),
            output_dir=Path(self.out_dir_var.get()),
            dpi=dpi_val,
            save_overlays=self.overlay_var.get(),
            save_contact_sheets=False,
            progress_callback=self._on_progress,
        )

        runner = SampleRunner(config=cfg, goal=goal)

        total = len(self.sample_rows_data)
        for idx, (sid, row_data) in enumerate(self.sample_rows_data.items(), 1):
            if self.stop_requested:
                break

            self.root.after(0, self._update_row_status, sid, "正在处理...")
            pct = int(((idx - 1) / total) * 100)
            self.root.after(0, self._set_progress, pct, f"[{idx}/{total}] 正在配准 {sid}...")

            # 注入用户在 GUI 中修改的镜像属性
            assets = row_data["assets"]
            assets.roi_evidence.is_mirrored = row_data["is_mirror"]

            try:
                report = runner.process(sid, assets=assets)
                status_str = report.get("overall_status", "PASS")
                self.root.after(0, self._update_row_status, sid, status_str)
            except Exception as e:
                self.root.after(0, self._update_row_status, sid, "失败")

        self.root.after(0, self._finish_batch)

    def _on_progress(self, sid: str, pct: int, msg: str):
        self.root.after(0, self.status_label_var.set, f"[{sid}] {msg}")

    def _set_progress(self, pct: float, msg: str):
        self.progress_var.set(pct)
        self.status_label_var.set(msg)

    def _update_row_status(self, sid: str, status_text: str):
        if sid in self.sample_rows_data:
            item_id = self.sample_rows_data[sid]["tree_item_id"]
            values = list(self.tree.item(item_id, "values"))
            values[6] = status_text
            self.tree.item(item_id, values=values)

    def _finish_batch(self):
        self.is_running = False
        self.progress_var.set(100.0)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label_var.set("批处理完成！点击【打开输出目录】查看结果。")
        messagebox.showinfo("完成", "所有样本配准提取处理完毕！")


def launch_gui():
    """GUI 启动入口函数"""
    root = tk.Tk()
    app = CrossStainWSIGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
