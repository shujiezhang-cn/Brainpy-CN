import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class CANN1D(bp.dyn.NeuDyn):
  def __init__(self, num, tau=1., k=0.4, a=0.7, A=10, J0=4,In=1,ns=1,
               z_min=-bm.pi, z_max=bm.pi):
    super(CANN1D, self).__init__(size=num)

    # parameters
    self.tau = tau  # The synaptic time constant
    self.k = k  # Degree of the rescaled inhibition
    self.a = a  # Half-width of the range of excitatory connections
    self.A = A  # Magnitude of the external input
    self.J0 = J0  # maximum connection value
    self.In = In # input noise strength
    self.ns = ns  # inner noise strength

    # feature space
    self.z_min = z_min
    self.z_max = z_max
    self.z_range = z_max - z_min
    self.x = bm.linspace(z_min, z_max, num)  # The encoded feature values
    self.rho = num / self.z_range  # The neural density
    self.dx = self.z_range / num  # The stimulus density

    # variables
    self.u = bm.Variable(bm.zeros(num))
    self.r = bm.Variable(bm.zeros(num))
    self.input = bm.Variable(bm.zeros(num))

    # The connection matrix
    self.conn_mat = self.make_conn(self.x)

    # function
    self.integral = bp.odeint(self.derivative)

  def derivative(self, u, t, Iext):
    r1 = bm.square(u)
    r2 = 1.0 + self.k * bm.sum(r1)
    r = r1 / r2
    Irec = bm.dot(self.conn_mat, r)
    du = (-u + Irec + Iext) / self.tau
    return du

  def dist(self, d):
    d = bm.remainder(d, self.z_range) #余数
    d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d) #距离过大的减去一个周期
    return d

  def make_conn(self, x):
    assert bm.ndim(x) == 1
    x_left = bm.reshape(x, (-1, 1))    #将数组变为一列
    x_right = bm.repeat(x.reshape((1, -1)), len(x), axis=0)  #创造XxX
    d = self.dist(x_left - x_right)
    Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a))/(bm.sqrt(2 * bm.pi) * self.a)
    return Jxx

  def get_stimulus_by_pos(self, pos):  #I_ext
    I_input = self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - pos) / self.a))
    return I_input

  def update(self):
    noise=self.ns*bm.random.randn(*bm.shape(self.u))* bm.sqrt(bp.share['dt'])
    intergral_u= self.integral(self.u, bp.share['t'], self.input, bp.share['dt'])
    intergral_u+=noise
    self.u.value=intergral_u
    r1 = bm.square(self.u)
    r2 = 1.0 + self.k * bm.sum(r1)
    r = r1 / r2
    self.r.value=r

  # 重置外界输入
    self.input[:] = 0.

cann = CANN1D(num=120)

#刺激位置
stimulus_position1=0.
stimulus_position2=1.5
I1 = cann.get_stimulus_by_pos(stimulus_position1) # N*1向量
I2 = cann.get_stimulus_by_pos(stimulus_position2)

# durations
(pre_stimulus_period, stimulus_period1, interval,
 stimulus_period2, post_stimulus_period)=5., 10., 10., 10., 10.

Iext_st, duration = bp.inputs.section_input(values=[0., I1, 0., I2, 0.],
                                         durations=[pre_stimulus_period, stimulus_period1, interval,
                                                    stimulus_period2,post_stimulus_period],
                                         return_length=True)

Iext_n1 = bp.inputs.wiener_process(duration,
                                  n=cann.num, t_start=pre_stimulus_period,
                                  t_end=pre_stimulus_period+stimulus_period1)
Iext_n2 = bp.inputs.wiener_process(duration,
                                  n=cann.num, t_start=pre_stimulus_period+stimulus_period1+interval,
                                  t_end=duration-post_stimulus_period)


Iext=cann.In*(Iext_n1+Iext_n2)+Iext_st

runner = bp.DSRunner(cann,
                     inputs=['input', Iext, 'iter'],
                     monitors=['u','r'])
runner.run(duration)


# 刺激位置神经元产生noisy bump
fig=plt.figure()
plt.plot(cann.x,Iext[int((pre_stimulus_period+stimulus_period1/2)/bp.share['dt']),:],label='Iext')
plt.plot(cann.x,runner.mon.u[int((pre_stimulus_period+stimulus_period1/2)/bp.share['dt']),:],label='u')
plt.xlabel('neuron',fontsize=16)
plt.title('noisy bump',fontsize=18)
plt.legend()


# population coding
ifpopulation1=bm.remainder(bm.abs(cann.x-stimulus_position1),cann.z_range)
index_population1=np.where(ifpopulation1<15*bm.pi/180)
u_population1=bm.mean(np.squeeze(runner.mon.u[:,index_population1]),axis=1)
r_population1=bm.mean(np.squeeze(runner.mon.r[:,index_population1]),axis=1)
Iext_population1=bm.mean(np.squeeze(Iext[:,index_population1]),axis=1)

ifpopulation2=bm.remainder(bm.abs(cann.x-stimulus_position2),cann.z_range)
index_population2=np.where(ifpopulation2<15*bm.pi/180)
u_population2=bm.mean(np.squeeze(runner.mon.u[:,index_population2]),axis=1)
r_population2=bm.mean(np.squeeze(runner.mon.r[:,index_population2]),axis=1)
Iext_population2=bm.mean(np.squeeze(Iext[:,index_population2]),axis=1)


#decoding
#time point
t1=int(pre_stimulus_period/bp.share['dt']) #开始展示刺激1的时刻对应的index
t2=int((pre_stimulus_period+stimulus_period1)/bp.share['dt'])   #刺激1展示结束时刻对应的index
t3=int((pre_stimulus_period+stimulus_period1+interval/2)/bp.share['dt']) #刺激1展示结束到开始展示刺激2的中间时刻对应的index
t4=int((duration-post_stimulus_period-stimulus_period2/2)/bp.share['dt']) #展示刺激2的中间时刻对应的index
t5=int((duration-post_stimulus_period/2)/bp.share['dt'])    #刺激2展示后的某一时刻对应的index

exppos=bm.exp(1j*cann.x)
mean_r_1=bm.mean(runner.mon.r[t1:t2,:],axis=0) #给stimulus期间
mean_r_2=bm.mean(runner.mon.r[t2+1:t3,:],axis=0) #interval
mean_r_3=bm.mean(runner.mon.r[t5:,:],axis=0) #撤掉刺激后
decode_p1=bm.angle(bm.dot(exppos,mean_r_1)) # 弧度
decode_p2=bm.angle(bm.dot(exppos,mean_r_2))
decode_p3=bm.angle(bm.dot(exppos,mean_r_3))


#神经元群响应
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax1.plot(runner.mon.ts, u_population1,label='group1')
ax1.plot(runner.mon.ts, u_population2,label='group2')
ax1.set_ylabel('synaptic current u',fontsize=16)
plt.title('population level',fontsize=18)
plt.legend()

ax2 = plt.subplot(gs[1])
ax2.plot(runner.mon.ts, Iext_population1,label='group1')
ax2.plot(runner.mon.ts, Iext_population2,label='group2')
ax2.set_ylabel('$I_{ext}$',fontsize=16)
ax2.set_xlabel('Time',fontsize=16)
plt.legend()


# decoing结果
fig, gs = plt.subplots(3, 1, figsize=(8, 8), sharex='all')
gs[0].plot(cann.x, Iext[int(t1/2+t2/2),:], label='Iext1')
gs[0].plot(cann.x, runner.mon.u[int(t1/2+t2/2),:], label='u(during stimulus1)')
gs[0].axvline(decode_p1,  linestyle='dashed', color=u'#444444',label='decoded position')
gs[0].set_title('decoding results',fontsize=18)
gs[0].legend()

gs[1].plot(cann.x, Iext[int(t1/2+t2/2),:], label='Iext1')
gs[1].plot(cann.x, runner.mon.u[int(t2/2+t3/2),:], label='u(during interval)')
gs[1].axvline(decode_p2, linestyle='dashed', color=u'#444444',label='decoded position')
gs[1].legend()

gs[2].plot(cann.x, Iext[t4,:], label='Iext2')
gs[2].plot(cann.x, runner.mon.u[int(t5/2+duration/bp.share['dt']/2),:], label='u(stimulus diminish)')
gs[2].axvline(decode_p3,  linestyle='dashed', color=u'#444444',label='decoded position')
gs[2].set_xlabel ('neuron',fontsize=16)
gs[2].legend()
plt.subplots_adjust(hspace=0.1)



#template matching
anim=bp.visualize.animate_1D(
  dynamical_vars=[{'ys': runner.mon.u, 'xs': cann.x, 'legend': 'u'},
                  {'ys': Iext, 'xs': cann.x, 'legend': 'Iext'}],
  frame_step=1, # 每个步长需要多少帧来显示
  frame_delay=0.01,  # 显示每一帧的delay
  show=True,
  # save_path='XXX/cann-encoding.gif' #存动图的路径
)
plt.show()