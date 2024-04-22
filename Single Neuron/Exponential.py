import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
class Exponential(bp.dyn.TwoEndConn):
    def __init__(self, pre, post, conn, g_max=0.02, tau=12., delay_step=2, E=0.,
        syn_type='CUBA', method='exp_auto', **kwargs):
        super(Exponential, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
        self.tau = tau
        self.g_max = g_max
        self.delay_step = delay_step
        self.E = E
        self.type = syn_type
        # 获取关于连接的信息
        self.pre2post = self.conn.require('pre2post')
        # 初始化变量
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)
        # 定义积分函数
        self.integral = bp.odeint(self.derivative, method=method)

    def derivative(self, g, t):
        dgdt = -g / self.tau
        return dgdt

    def update(self, tdi):
        # 取出延迟了delay_step时间步长的突触前脉冲信号
        delayed_pre_spike = self.delay(self.delay_step)
        self.delay.update(self.pre.spike)
        # 根据连接模式计算各个突触后神经元收到的信号强度
        post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post,
                                        self.post.num, self.g_max)
        # 突触的电导g的更新包括常规积分和突触前脉冲带来的跃变
        self.g.value = self.integral(self.g, tdi.t, tdi.dt) + post_sp
        # 计算突触后电流
        if self.type == 'CUBA':
            self.post.input += self.g * (self.E - (-65.)) # E - V_rest

def run_syn(syn_model, title, run_duration=200., sp_times=(10, 20, 30), **kwargs):
    # 定义突触前神经元、突触后神经元和突触连接，并构建神经网络
    neu1 = bp.neurons.SpikeTimeGroup(1, times=sp_times, indices=[0] * len(sp_times))
    neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
    syn1 = syn_model(neu1, neu2, conn=bp.connect.All2All(), **kwargs)
    net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)
    runner = bp.dyn.DSRunner(net, monitors=['pre.spike', 'post.V', 'syn.g', 'post.input'])
    runner.run(run_duration)
    fig, gs = bp.visualize.get_figure(7, 1, 0.5, 6.)
    ax = fig.add_subplot(gs[0, 0])
    plt.plot(runner.mon.ts, runner.mon['pre.spike'], label='pre.spike')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xticks([])
    ax = fig.add_subplot(gs[1:3, 0])
    plt.plot(runner.mon.ts, runner.mon['syn.g'], label='g', color=u'#d62728')
    plt.legend(loc='upper right')
    plt.xticks([])
    ax = fig.add_subplot(gs[3:5, 0])
    plt.plot(runner.mon.ts, runner.mon['post.input'], label='PSC', color=u'#d62728')
    plt.legend(loc='upper right')
    plt.xticks([])
    ax = fig.add_subplot(gs[5:7, 0])
    plt.plot(runner.mon.ts, runner.mon['post.V'], label='post.V')
    plt.legend(loc='upper right')
    plt.xlabel('t[ms]')
    plt.show()

run_syn(Exponential,sp_times=[25, 50, 75, 100, 160],
title='Exponential Synapse Model (Current-Based)',
syn_type='CUBA', )