"""
Script to obtain the linear time simulation of a State Space Model
representing a one-stage CCCP
It includes capacitive-resistive load.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv

def modelParams(Tc, c1, cp, cl, rs, rl, vin, vck):
    # Effect of parasitic caps
    vckp = vck * c1 / (c1 + cp)

    # Derived constants
    tau1 = rs * (c1 + cp)
    tauL = rl * cl
    tau3 = rs * cl

    # Constants for 1st stage equation
    # Vi' = h1*Vi + h2*Vin
    h1 = np.exp(-Tc / (2 * tau1))
    h2 = 1.0 - h1

    # Constants for charge redistribution equation
    sr1 = -((1 / tauL) + (1 / tau3) + (1 / tau1)) / 2
    sr2 = (1 / (4 * tau1 * tau1)) + (1 / (4 * tau3 * tau3)) + (1 / (4 * tauL * tauL)) + (1 / (2 * tau1 * tau3)) \
          + (1 / (2 * tau3 * tauL)) - (1 / (4 * tau1 * tauL))
    s1 = (sr1 + np.sqrt(sr2))
    s2 = (sr1 - np.sqrt(sr2))
    difs = 2 * sr1
    # print("s1, s2: ", s1, s2)

    exp_s1 = np.exp(s1 * Tc / 2)
    exp_s2 = np.exp(s2 * Tc / 2)

    # VL' = g1*Vi + g1*Vck + g2*VL
    g1 = (exp_s2 - exp_s1) / (tau3 * difs)
    g2 = (exp_s1 * ((1 / tau3) + (1 / tauL) + s2) - exp_s2 * ((1 / tau3) + (1 / tauL) + s1)) / difs
    # print("g factors: ", g1, g2)

    # V2' = ft1*V2 + ft2*VL + ft3*Vck
    r1 = ((1 - exp_s1) / s1 + (exp_s2 - 1) / s2) / (tauL * difs)
    rcf2 = ((1 / tau3) + (1 / tauL))
    rcfs1 = rcf2 + s2
    rcfs2 = rcf2 + s1
    r2 = (((1 - exp_s1) * rcfs1 / s1) + ((1 - exp_s2) * rcfs2 / s2)) / difs

    ft1 = (1 - (r1 / tau1) - (tau3 * g1 / tau1))
    ft2 = ((1 - g2) * tau3 / tau1) - (r2 * tau3 / (tauL * tau1))
    ft3 = -(tau3 * g1 / tau1) - (r1 / tau1)
    # print("fts:", ft1, ft2, ft3)

    # Matrius per valors de semiperiodes
    Asp1 = [[h1, 0.0, 0.0],
            [0.0, ft1, ft2],
            [0.0, g1, g2]]

    Bsp1 = [[h2 * vin],
            [ft3 * vckp],
            [g1 * vckp]]

    Asp2 = [[ft1, 0.0, ft2],
            [0.0, h1, 0.0],
            [g1, 0.0, g2]]

    Bsp2 = [[ft3 * vckp],
            [h2 * vin],
            [g1 * vckp]]

    # Matrix A
    A = [[(h1 * ft1), (g1 * ft2), (g2 * ft2)],
         [0.0, (h1 * ft1), (h1 * ft2)],
         [(g1 * h1), (g2 * g1), (g2 * g2)]]

    # Matrix B
    B = [[(h2 * ft1 * vin) + ((ft2 * g1) + ft3) * vckp],
         [(h2 * vin) + (h1 * ft3 * vckp)],
         [(g1 * h2 * vin) + (g1 * (1 + g2) * vckp)]]

    # print("B: ", B)
    # Matrix C - output
    C = [0.0, 0.0, 1.0]
    D = [0.0]


    # Results
    # eigen_A = np.linalg.eig(A)[0]

    # print("Eigenvalues", eigen_A)

    # sys1 = signal.StateSpace(A, B, C, D)
    # sys1 = signal.dlti(A, B, C, D, dt=Tc)

    # t1, y1 = signal.step(sys1)
    # tsamples = np.linspace(0.0, 150*Tc, 151)
    # u = np.zeros(len(tsamples))
    # u[1:] = 1.0
    # t1, y1 = signal.dstep(sys1, t=tsamples)
    # t1, y1, x1 = signal.dlsim(sys1, u, t)


    def seq_whole_Tc():
        v1seq = [0.0]
        v2seq = [0.0]
        vLseq = [0.0]
        i = 0
        td = [0]
        diff1 = 10.0
        diff2 = 10.0
        diffL = 10.0
        limit = 0.0000001 * (vin + vck)

        while diffL > limit:
            newv1 = A[0][0] * v1seq[i] + A[0][1] * v2seq[i] + A[0][2] * vLseq[i] + B[0][0]
            newv2 = A[1][0] * v1seq[i] + A[1][1] * v2seq[i] + A[1][2] * vLseq[i] + B[1][0]
            newvL = A[2][0] * v1seq[i] + A[2][1] * v2seq[i] + A[2][2] * vLseq[i] + B[2][0]
            # print(newvL[0])
            diff1 = newv1 - v1seq[i]
            diff2 = newv2 - v2seq[i]
            diffL = newvL - vLseq[i]
            v1seq.append(newv1)
            v2seq.append(newv2)
            vLseq.append(newvL)
            i += 1
            td.append(i * Tc)
        return [i, td, vLseq, v1seq]


    def seq_half_Tc():
        v1seq = [0.0]
        v2seq = [0.0]
        vLseq = [0.0]
        i = 0
        td = [0]
        diff1 = 10.0
        diff2 = 10.0
        diffL = 10.0
        limit = 0.0000001 * (vin + vck)

        while diffL > limit:
            newv1p = Asp1[0][0] * v1seq[i] + Asp1[0][1] * v2seq[i] + Asp1[0][2] * vLseq[i] + Bsp1[0][0]
            newv2p = Asp1[1][0] * v1seq[i] + Asp1[1][1] * v2seq[i] + Asp1[1][2] * vLseq[i] + Bsp1[1][0]
            newvLp = Asp1[2][0] * v1seq[i] + Asp1[2][1] * v2seq[i] + Asp1[2][2] * vLseq[i] + Bsp1[2][0]
            v1seq.append(newv1p)
            v2seq.append(newv2p)
            vLseq.append(newvLp)
            i += 1
            td.append(i * Tc / 2)
            newv1pp = Asp2[0][0] * v1seq[i] + Asp2[0][1] * v2seq[i] + Asp2[0][2] * vLseq[i] + Bsp2[0][0]
            newv2pp = Asp2[1][0] * v1seq[i] + Asp2[1][1] * v2seq[i] + Asp2[1][2] * vLseq[i] + Bsp2[1][0]
            newvLpp = Asp2[2][0] * v1seq[i] + Asp2[2][1] * v2seq[i] + Asp2[2][2] * vLseq[i] + Bsp2[2][0]
            v1seq.append(newv1pp)
            v2seq.append(newv2pp)
            vLseq.append(newvLpp)
            i += 1
            td.append(i * Tc / 2)
            diffL = newvLpp - vLseq[i - 2]
        return [i, td, vLseq, v1seq, v2seq]


    kss, tdss, vl_ss, v1_ss = seq_whole_Tc()
    # print("steady state voltage (whole): ", kss, tdss[kss], vl_ss[kss])

    ksp, tdsp, vl_sp, v1_sp, v2_sp = seq_half_Tc()
    # print("steady state voltage (half): ", ksp, tdsp[ksp], vl_sp[ksp])

    # Calcul d'amplitud de ripple
    # Es calcula el punt de treball partint del steady state.

    fa = (-(v1_ss[kss] + vck) / tau3 + vl_ss[kss] * ((1 / tau3) + (1 / tauL) + s2)) / difs
    fb = ((v1_ss[kss] + vck) / tau3 - vl_ss[kss] * ((1 / tau3) + (1 / tauL) + s1)) / difs

    def plotTrans():
        tcad = []
        vcad = []
        first = 1
        rowcount = 0

        with open('CP_V350m_CF6f_TC2n.csv', 'r') as csvfile:
            cadplots = csv.reader(csvfile, delimiter=',')
            for row in cadplots:
                if (first == 1):
                    first = 0
                else:
                    if rowcount == 5:
                        rowcount = 0
                        tcad.append(float(row[0]))
                        vcad.append(float(row[1]))
                    else:
                        rowcount += 1

        from matplotlib.ticker import StrMethodFormatter
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0e}'))  # 0 decimal place

        plt.title('Vin=350mV; Cf=6fF')
        plt.plot(tcad, vcad, label='Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Load Voltage (V)')

        # plt.step(t1, y1[0])
        plt.step(tdsp, vl_sp, label='Model')
        # plt.step(tdss, v1_ss)
        # plt.step(tdsp, v1_sp)
        plt.legend()
        plt.show()

    def derivVL(x):
        if x < 0:
            out = derivVL(0)
        elif x > 0.5:
            out = derivVL(0.5)
        else:
            out = fa * s1 * np.exp(s1 * x * Tc / 2) + fb * s2 * np.exp(s2 * x * Tc / 2)
        return out


    def VL(x):
        return fa * np.exp(s1 * x) + fb * np.exp(s2 * x)



    # print(derivVL(0.23))
    # print(VL(0.6*Tc/2)-VL(0))


    def findMaxVL(limitx, limity):
        diferx = 1.0
        difery = 1e9
        xlow = 0
        xhigh = 0.5
        ylow = derivVL(xlow)
        yhigh = derivVL(xhigh)
        newx = 0
        newy = 0
        if yhigh > 0:
            return [0.0, 0.0]
        else:
            while (diferx > limitx) | (difery > limity):
                newx = (xhigh * ylow - xlow * yhigh) / (ylow - yhigh)
                newy = derivVL(newx)
                if newy > 0:
                    diferx = newx - xlow
                    difery = newy
                    xlow = newx
                    ylow = newy
                else:
                    diferx = xhigh - newx
                    difery = -newy
                    xhigh = newx
                    yhigh = newy
            return [newx, newy]


    xm, ym = findMaxVL(0.01, 50.0)

    ripple = VL(xm * Tc / 2) - VL(0)
    # print("tmax, ripple:", xm, ripple)

    # plotTrans()

    return ksp, tdsp[ksp], vl_sp[ksp], ripple



# input data
vin = 0.35
vck = 0.35

rs = 200e3
c1 = [6e-15, 20e-15, 60e-15]
cp = 3.1e-15

rl = 100e6
cl = 1e-12

Tc = [2e-9, 10e-9, 100e-9, 500e-9, 1000e-9, 3000e-9]

for cfly in c1:
    ssv_tc = []
    rip_tc = []
    stm_tc = []
    for tclock in Tc:
        steps, tss, ssv, vr = modelParams(tclock, cfly, cp, cl, rs, rl, vin, vck)
        ssv_tc.append(ssv)
        rip_tc.append(vr)
        stm_tc.append(tss)
    plegend = "Cf=", cfly
    plt.plot(Tc, ssv_tc, label=plegend)

from matplotlib.ticker import StrMethodFormatter

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1e}'))  # 1 decimal place

plt.title('Ripple Voltage')
plt.xlabel('Clock period (s)')
plt.ylabel('Ripple Voltage (V)')
plt.legend()
plt.show()
