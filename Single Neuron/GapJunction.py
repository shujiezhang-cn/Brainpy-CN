import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
class GapJunction(bp.dyn.TwoEndConn):
    def __init__(self, pre, post, conn, g=0.2, **kwargs):
        super(GapJunction, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
        # 初始化参数
        self.g = g
        # 获取每个连接的突触前神经元pre_ids和突触后神经元post_ids
        self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
        # 初始化变量
        self.current = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi):
        # 计算突触后电流（从外向内为正方向）
        inputs = self.g * (self.pre.V[self.pre_ids] - self.post.V[self.post_ids])
        # 从synapse到post的计算：post id相同电流加和到一起
        self.current.value = bm.syn2post(inputs, self.post_ids, self.post.num)
        self.post.input += self.current
def run_syn_GJ(syn_model, title, run_duration=100., Iext=7.5, **kwargs):
# 定义神经元组和突触连接，并构建神经网络
    neu = bp.neurons.HH(2, V_initializer=bp.init.Constant(-70.68))
    syn = syn_model(neu, neu, conn=bp.connect.All2All(include_self=False), **kwargs)
    net = bp.dyn.Network(syn=syn, neu=neu) # include_self=False: 自己和自己没有连接
    # 运行模拟
    runner = bp.dyn.DSRunner(net,inputs=[('neu.input', bm.array([Iext, 0.]))],
                         monitors=['neu.V', 'syn.current'],jit=True)

    runner.run(run_duration)
    # 可视化
    fig, gs = plt.subplots(2, 1, figsize=(6, 4.5))
    plt.sca(gs[0])
    plt.plot(runner.mon.ts, runner.mon['neu.V'][:, 0], label='neu0-V')
    plt.plot(runner.mon.ts, runner.mon['neu.V'][:, 1], label='neu1-V')
    plt.legend(loc='upper right')
    plt.title(title)

    plt.sca(gs[1])
    plt.plot(runner.mon.ts, runner.mon['syn.current'][:, 0],
    label='neu1-current', color=u'#48d688')
    plt.plot(runner.mon.ts, runner.mon['syn.current'][:, 1],
    label='neu0-current', color=u'#d64888')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

run_syn_GJ(GapJunction, Iext=7.5, title='Gap Junction Model')
run_syn_GJ(GapJunction, Iext=5., title='Gap Junction Model')