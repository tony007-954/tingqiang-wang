# -*- coding: utf-8 -*-
import wx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import fatpack
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

class SNCurveMMPDS:
    def __init__(self, A=11.8, B=4.38, C=12.0, m=0.61):
        self.A = A
        self.B = B
        self.C = C
        self.m = m
    
    def S_eq(self, S_max, R):
        return S_max * (1 - R)**self.m
    
    def fatigue_life(self, S_max, R):
        seq = self.S_eq(S_max, R)
        if np.any(seq <= self.C):
            return np.full_like(seq, np.nan)
        logNf = self.A - self.B * np.log10(seq - self.C)
        Nf = 10**logNf
        return Nf

class MyFrame(wx.Frame):
    def __init__(self, parent, title="Fatigue Analysis GUI"):
        super(MyFrame, self).__init__(parent, title=title, size=(1000, 700))
        
        self.data = None  # 原始数据
        self.x_data = None
        self.y_data = None
        self.ranges = None
        self.N = None
        self.S = None
        self.sample_freq = 4096.0

        # 新增的成员变量用于存储过滤后的N,S数据
        self.filtered_N = None
        self.filtered_S = None

        self.init_ui()

    def init_ui(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        load_btn = wx.Button(panel, label="加载数据(TXT)")
        load_btn.Bind(wx.EVT_BUTTON, self.on_load_data)
        hbox1.Add(load_btn, 0, wx.ALL, 5)
        
        show_btn = wx.Button(panel, label="显示折线图")
        show_btn.Bind(wx.EVT_BUTTON, self.on_show_line_plot)
        hbox1.Add(show_btn, 0, wx.ALL, 5)
        
        show_fatigue_btn = wx.Button(panel, label="显示疲劳分析图")
        show_fatigue_btn.Bind(wx.EVT_BUTTON, self.on_show_plots)
        hbox1.Add(show_fatigue_btn, 0, wx.ALL, 5)
        
        miner_btn = wx.Button(panel, label="Miner寿命估计(原始)")
        miner_btn.Bind(wx.EVT_BUTTON, self.on_miner_estimate)
        hbox1.Add(miner_btn, 0, wx.ALL, 5)

        # 新增按钮：加载filtered_cycles.txt并进行疲劳分析
        load_filtered_btn = wx.Button(panel, label="加载过滤数据并分析")
        load_filtered_btn.Bind(wx.EVT_BUTTON, self.on_load_filtered_and_analyze)
        hbox1.Add(load_filtered_btn, 0, wx.ALL, 5)

        vbox.Add(hbox1, 0, wx.EXPAND|wx.ALL, 5)
        
        self.plot_panel = wx.Panel(panel)
        vbox.Add(self.plot_panel, 1, wx.EXPAND|wx.ALL, 5)
        
        panel.SetSizer(vbox)
        self.Centre()

    def on_load_data(self, event):
        """从TXT文件导入原始数据"""
        with wx.FileDialog(self, "选择TXT文件", wildcard="TXT files (*.txt)|*.txt",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # 用户取消
            
            pathname = fileDialog.GetPath()
            try:
                df = pd.read_csv(pathname, header=None, delim_whitespace=True)
                if df.shape[1] < 2:
                    wx.MessageBox("TXT文件数据列不足两列，请检查数据格式。", "错误", wx.ICON_ERROR)
                    return
                
                self.x_data = df.iloc[:, 0].values
                self.y_data = df.iloc[:, 1].values

                # 雨流计数分析
                y = self.y_data
                reversals, reversals_ix = fatpack.find_reversals(y)
                cycles, residue = fatpack.find_rainflow_cycles(reversals)
                processed_residue = fatpack.concatenate_reversals(residue, residue)
                cycles_residue, _ = fatpack.find_rainflow_cycles(processed_residue)
                cycles_total = np.concatenate((cycles, cycles_residue))
                self.ranges = np.abs(cycles_total[:, 1] - cycles_total[:, 0])
                
                self.data = y
                wx.MessageBox("数据加载成功！", "信息", wx.ICON_INFORMATION)
            except Exception as e:
                wx.MessageBox(f"加载数据失败: {e}", "错误", wx.ICON_ERROR)
    
    def on_show_line_plot(self, event):
        """显示时域数据折线图"""
        if self.x_data is None or self.y_data is None:
            wx.MessageBox("请先加载数据！", "错误", wx.ICON_ERROR)
            return
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=96)
        ax.plot(self.x_data, self.y_data, marker='o', linestyle='-', label='Time-Stress Data')
        
        x_min, x_max = np.min(self.x_data), np.max(self.x_data)
        y_min, y_max = np.min(self.y_data), np.max(self.y_data)
        
        x_margin = (x_max - x_min)*0.05 if (x_max - x_min)!=0 else 1.0
        y_margin = (y_max - y_min)*0.05 if (y_max - y_min)!=0 else 1.0
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Stress")
        ax.set_title("Line Plot (Time vs Stress)")
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def on_show_plots(self, event):
        """显示雨流分析，FFT和S-N曲线等图"""
        if self.data is None:
            wx.MessageBox("请先加载数据！", "错误", wx.ICON_ERROR)
            return
        
        y = self.data
        fig = plt.figure(figsize=(14,10), dpi=96)
        
        # 子图1：Signal with reversals
        ax_signal = fig.add_subplot(3, 3, 1)
        reversals_ix = fatpack.find_reversals(y)[1]
        ax_signal.plot(y)
        ax_signal.plot(reversals_ix, y[reversals_ix], 'ro', fillstyle='none', label='reversal')
        ax_signal.legend()
        ax_signal.set_title("Signal")
        ax_signal.set_ylabel("y")
        ax_signal.set_xlabel("Index")
        
        # 子图2：Cumulative distribution
        ax_cumdist = fig.add_subplot(3, 3, 4)
        N, S = fatpack.find_range_count(self.ranges, 64)
        Ncum = N.sum() - np.cumsum(N)
        ax_cumdist.semilogx(Ncum, S)
        ax_cumdist.set_title("Cumulative distribution, rainflow ranges")
        ax_cumdist.set_xlabel("Count, N")
        ax_cumdist.set_ylabel("Range, S")
        
        self.N = N
        self.S = S

        # 子图3：Rainflow matrix 2D
        ax_rfcmat = fig.add_subplot(3, 3, 2, aspect='equal')
        reversals, reversals_ix = fatpack.find_reversals(y)
        cycles, residue = fatpack.find_rainflow_cycles(reversals)
        processed_residue = fatpack.concatenate_reversals(residue, residue)
        cycles_residue, _ = fatpack.find_rainflow_cycles(processed_residue)
        cycles_total = np.concatenate((cycles, cycles_residue))

        bins = np.linspace(cycles_total.min(), cycles_total.max(), 64)
        rfcmat = fatpack.find_rainflow_matrix(cycles_total, bins, bins)
        x_centers = (bins[:-1] + bins[1:]) / 2.0
        X, Y = np.meshgrid(x_centers, x_centers, indexing='ij')
        C = ax_rfcmat.pcolormesh(X, Y, rfcmat, cmap='magma')
        fig.colorbar(C, ax=ax_rfcmat)
        ax_rfcmat.set_title("Rainflow matrix (2D)")
        ax_rfcmat.set_xlabel("Starting point")
        ax_rfcmat.set_ylabel("Destination point")
        
        # 子图4：3D雨流图
        ax_3d = fig.add_subplot(3, 3, 3, projection='3d')
        ax_3d.plot_surface(X, Y, rfcmat, cmap='viridis', edgecolor='none')
        ax_3d.set_title("Rainflow matrix (3D)")
        ax_3d.set_xlabel("Starting point")
        ax_3d.set_ylabel("Destination point")
        ax_3d.set_zlabel("Counts")
        
        # 子图5：S-N曲线
        ax_sn = fig.add_subplot(3, 3, 5)
        curve = fatpack.TriLinearEnduranceCurve(160)
        N_range = np.logspace(6, 9)
        S_range = curve.get_stress(N_range)
        ax_sn.loglog(N_range, S_range)
        ax_sn.set(xlim=(1e6, 2e8), ylim=(1., 1000),
                  title="Endurance curve, detail category 160 Mpa",
                  xlabel="Endurance [1]", ylabel="Stress Range [Mpa]")
        ax_sn.grid()
        ax_sn.grid(which='both')

        # 子图6：FFT频域图
        ax_fft = fig.add_subplot(3, 3, 6)
        Y_fft = np.fft.fft(y)
        N_points = len(y)
        freqs = np.fft.fftfreq(N_points, d=1.0/self.sample_freq)
        positive_mask = freqs >= 0
        freqs_pos = freqs[positive_mask]
        Y_pos = Y_fft[positive_mask]
        amplitude = np.abs(Y_pos)
        
        ax_fft.semilogx(freqs_pos, amplitude)
        ax_fft.set_title("Frequency Domain (FFT)")
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Amplitude")
        ax_fft.grid(True)
        
        plt.tight_layout()
        plt.show()

    def on_miner_estimate(self, event):
        """对原始数据的雨流统计（N,S）进行Miner法计算"""
        if self.N is None or self.S is None:
            wx.MessageBox("请先显示疲劳分析图以计算N,S！", "错误", wx.ICON_ERROR)
            return
        
        curve = fatpack.TriLinearEnduranceCurve(160)
        
        D = 0.0
        for n_i, S_i in zip(self.N, self.S):
            N_i = curve.get_endurance(S_i)
            D += n_i / N_i

        msg = f"Cumulative Damage (D): {D}\n"
        if D >= 1.0:
            msg += "Predicted fatigue failure at the given loading condition."
        else:
            msg += "Remaining fatigue life available."
        wx.MessageBox(msg, "Miner Result", wx.ICON_INFORMATION)

    def on_load_filtered_and_analyze(self, event):
        """加载filtered_cycles.txt的数据进行S-N分析和Miner计算"""
        filtered_path = 'filtered_cycles.txt'  # 请根据实际情况修改路径
        
        try:
            # 假设filtered_cycles.txt中每行两列[start, end]
            fc = np.loadtxt(filtered_path)
            if fc.shape[1] < 2:
                wx.MessageBox("filtered_cycles.txt数据格式有误，请检查。", "错误", wx.ICON_ERROR)
                return
            
            # 计算ranges
            ranges_filtered = np.abs(fc[:,1] - fc[:,0])
            # rainflow range count
            Nf, Sf = fatpack.find_range_count(ranges_filtered, 64)
            self.filtered_N = Nf
            self.filtered_S = Sf

            # 使用TriLinearEnduranceCurve(160)进行Miner计算
            curve = fatpack.TriLinearEnduranceCurve(160)
            D_filtered = 0.0
            for n_i, S_i in zip(self.filtered_N, self.filtered_S):
                N_i = curve.get_endurance(S_i)
                D_filtered += n_i / N_i

            msg = f"Filtered Data Cumulative Damage (D): {D_filtered}\n"
            if D_filtered >= 1.0:
                msg += "Predicted fatigue failure at given conditions (filtered data)."
            else:
                msg += "Remaining fatigue life available (filtered data)."

            wx.MessageBox(msg, "Filtered Data Miner Result", wx.ICON_INFORMATION)

        except Exception as e:
            wx.MessageBox(f"加载filtered_cycles.txt数据失败: {e}", "错误", wx.ICON_ERROR)

class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame(None, "Fatigue Analysis GUI")
        frame.Show()
        return True

if __name__ == "__main__":
    app = MyApp(False)
    app.MainLoop()
