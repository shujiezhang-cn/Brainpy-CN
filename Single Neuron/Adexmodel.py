import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
class AdEx(bp.dyn.NeuGroup):
        def __init__(self, size, V_rest=-70., V_reset=-68., V_th=0., V_T=-50.
                     , delta_T=2., a=1.,b=2.5, R=0.5, tau=10.,
                     tau_w=30., tau_ref=0., name=None):
            super(AdEx, self).__init__(size=size, name=name)
            self.V_rest = V_rest
            self.V_reset = V_reset
            self.V_th = V_th
            self.V_T = V_T
            self.delta_T = delta_T
            self.a = a
            self.b = b
            self.tau = tau
            self.tau_w = tau_w
            self.R = R
            self.tau_ref = tau_ref

            self.V = bm.Variable(bm.ones(self.num) * V_rest)
            self.w = bm.Variable(bm.zeros(self.num))
            self.input = bm.Variable(bm.zeros(self.num))
            self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
            self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
            self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))

# 定义积分器
            self.integral = bp.odeint(f=self.derivative, method='exp_auto')

        def dV(self, V, t, w, Iext):
            _tmp = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
            dVdt = (- V + self.V_rest + _tmp - self.R * w + self.R * Iext) / self.tau
            return dVdt

        def dw(self, w, t, V):
            dwdt = (self.a * (V - self.V_rest) - w) / self.tau_w
            return dwdt

# 将两个微分方程联合为一个，以便同时积分
        @property
        def derivative(self):
            return bp.JointEq(self.dV, self.dw)

        def update(self, tdi):
            _t, _dt = tdi.t, tdi.dt
            V, w = self.integral(self.V, self.w, _t, self.input, dt=_dt)
            refractory = (_t - self.t_last_spike) <= self.tau_ref
            V = bm.where(refractory, self.V, V)
            spike = V > self.V_th
            self.spike.value = spike
            self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
            self.V.value = bm.where(spike, self.V_reset, V)
            self.w.value = bm.where(spike, w + self.b, w)
            self.refractory.value = bm.logical_or(refractory, spike)
            self.input[:] = 0.


mode='ADAPTATION'
if mode=='TONIC':
    tau = 20
    tau_w = 30
    a = 0
    b = 60
    V_reset = -55
    I = 65

if mode=='ADAPTATION':
    tau = 20
    tau_w = 100
    a = 0
    b = 5
    V_reset = -55
    I = 65

if mode=='INITIAL':
    tau = 5
    tau_w = 100
    a = 0.5
    b = 7
    V_reset = -51
    I = 65
if mode=='BURSTING':
    tau = 5
    tau_w = 100
    a = -0.5
    b = 7
    V_reset = -47
    I = 65
if mode=='TRANSIENT':
    tau = 10
    tau_w = 100
    a = 1
    b = 10
    V_reset = -60
    I = 55
if mode=='DELAYED':
    tau = 5
    tau_w = 100
    a = -1
    b = 5
    V_reset = -60
    I = 25
    
neu = AdEx(1,tau = tau,tau_w = tau_w,a = a,b = b,V_reset = V_reset)
runner = bp.dyn.DSRunner(neu, monitors=['V', 'w', 'spike'], inputs=('input', I))
runner(500)
# 可视化V和w的变化
runner.mon.V = np.where(runner.mon.spike, 20., runner.mon.V)
plt.plot(runner.mon.ts, runner.mon.V, label='V')
plt.plot(runner.mon.ts, runner.mon.w, label='w')
plt.xlabel('t (ms)')
plt.ylabel('V (mV)')

plt.show()