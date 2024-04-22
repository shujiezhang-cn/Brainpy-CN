import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
class VoltageJump(bp.dyn.TwoEndConn):
    def __init__(self, pre, post, conn, g_max=1., delay_step=0, E=0., **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, **kwargs)
        # 初始化参数
        self.g_max = g_max
        self.delay_step = delay_step
        self.E = E
        # 获取关于连接的信息
        self.pre2post = self.conn.require('pre2post')
        # 初始化变量
        self.g = bm.Variable(bm.zeros(self.post.num))
        self.delay = bm.LengthDelay(self.pre.spike, delay_step)

    def update(self, tdi):
        # 取出延迟了delay_step时间步长的突触前脉冲信号
        delayed_pre_spike = self.delay(self.delay_step)
        # 根据最新的突触前脉冲信号更新延迟变量
        self.delay.update(self.pre.spike)
        # 根据连接模式计算各个突触后神经元收到的信号强度
        post_sp = bm.pre2post_event_sum(delayed_pre_spike, self.pre2post,
                                        self.post.num, self.g_max)
        self.g.value = post_sp
        # 计算突触后效应
        self.post.V += self.g



neu1 = bp.neurons.SpikeTimeGroup(1,
times=[20, 60, 100, 140, 180],
indices=[0, 0, 0, 0, 0])
neu2 = bp.neurons.HH(1, V_initializer=bp.init.Constant(-70.68))
syn1 = VoltageJump(neu1, neu2, conn=bp.connect.All2All(), g_max=2.)
net = bp.dyn.Network(pre=neu1, syn=syn1, post=neu2)

runner = bp.dyn.DSRunner(net, monitors=['pre.spike', 'post.V', 'syn.g'])
runner.run(200)

# 可视化
fig, gs = bp.visualize.get_figure(3, 1, 1.5, 6.)
ax = fig.add_subplot(gs[0, 0])
plt.plot(runner.mon.ts, runner.mon['pre.spike'], label ='pre.spike')
plt.legend(loc='upper right')
plt.title('Delta synapse model')
plt.xticks([])
ax = fig.add_subplot(gs[1, 0])
plt.plot(runner.mon.ts, runner.mon['syn.g'], label ='g', color = u'#d62728')
plt.legend(loc='upper right')
plt.xticks([])
ax = fig.add_subplot(gs[2, 0])
plt.plot(runner.mon.ts, runner.mon['post.V'], label ='post.V')
plt.legend(loc='upper right')
plt.xlabel('t[ms]')
plt.show()