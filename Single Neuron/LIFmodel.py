import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, V_rest=0., V_reset=-5., V_th=20.,
                 R=1., tau=10., t_ref=5., name=None):
        super(LIF, self).__init__(size=size, name=name)
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.R = R
        self.tau = tau
        self.t_ref = t_ref

        self.V = bm.Variable(bm.ones(self.num) * V_rest)
        self.input = bm.Variable(bm.zeros(self.num))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
        self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

         # 使用指数欧拉方法进行积分
        self.integral = bp.odeint(f=self.derivative, method='exp_auto')

 # 定义膜电位关于时间变化的微分方程
    def derivative(self, V, t, R, Iext):
        dvdt = (-V + self.V_rest + R * Iext) / self.tau
        return dvdt

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        refractory = (t - self.t_last_spike) <= self.t_ref
        V = self.integral(self.V, t, self.R, self.input, dt=dt)
        V = bm.where(refractory, self.V, V)
        spike = V > self.V_th
        self.spike.value = spike
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.refractory.value = bm.logical_or(refractory, spike)
        self.input[:] = 0.




# 在恒定电流输入下，lifmodel以固定频率发放
currents1, length = bp.inputs.section_input(values=[0.,21,],
                                            durations=[50,150 ]
                                            ,return_length=True)
group = LIF(1)
runner1 = bp.dyn.DSRunner(group, monitors=['V'],
                          inputs=['input', currents1,'iter'])
runner1(length)
fig,axe=plt.subplots(2,1,gridspec_kw={'height_ratios':[2,1]})
axe[0].plot(runner1.mon.ts, runner1.mon.V,color='blue')
axe[1].plot(runner1.mon.ts, currents1,linewidth=2,color='blue')
plt.show()

# lif神经元发放率与电流的关系
# duration=1000
# I=np.arange(0,600,1)
# group=LIF(len(I))
# runner = bp.dyn.DSRunner(group,
#                          monitors=['spike'], inputs=['input', I])
# runner(duration=duration)
# F=runner.mon.spike.sum(axis=0)/(duration/1000)
# plt.plot(I,F,linewidth=2)
# plt.show()



##lifmodel的filter作用
# import math
# in_va=np.arange(0,100,0.1)
# duration=np.ones(len(in_va))*0.1
# def sin_value(I):
#     sin=[40*math.sin(i)+20 for i in I]
#     return sin
# value=sin_value(in_va)
# currents1, length = bp.inputs.section_input(values=value,
#                                             durations=duration
#                                             ,return_length=True)
# group = LIF(1)
# runner1 = bp.dyn.DSRunner(group, monitors=['V'],
#                           inputs=['input', currents1,'iter'])
# runner1(length)
# fig,axe=plt.subplots(2,1,gridspec_kw={'height_ratios':[2,1]})
# axe[0].plot(runner1.mon.ts, runner1.mon.V,color='blue')
# axe[1].plot(in_va,currents1,linewidth=2,color='blue')
# axe[1].set_xlabel('t (ms)')
# plt.show()