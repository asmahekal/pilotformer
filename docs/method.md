# Method and experiments

## System model

A single-antenna OFDM slot has $K = 72$ sub-carriers ($\Delta f = 15$ kHz) and $L = 14$ symbols. With inter-carrier
interference neglected ($f_D/\Delta f \le 0.067$), the received symbol on resource element $(k, l)$ is

$$Y[k,l] = H[k,l]\,X[k,l] + N[k,l], \qquad N \sim \mathcal{CN}(0, \sigma^2).$$

The channel is a tapped delay line whose tap gains are sums of 16 sinusoids (Jakes spectrum)
(`pilotformer.channels.TDLChannel`). Pilots carry $X = 1$, so the LS estimate on pilot $p$ is $\hat h_p = Y[k_p, l_p]$.

## Robust-LMMSE filter bank

For a design $(\tau_{\max}, f_D)$ the covariance is $\mathbf R = \mathbf R_f \otimes \mathbf R_t$ with

$$[\mathbf R_f]_{k,k'} = \frac{1 - e^{-j2\pi(k-k')\Delta f \tau_{\max}}}{j2\pi(k-k')\Delta f\tau_{\max}}, \qquad
[\mathbf R_t]_{l,l'} = J_0\big(2\pi f_D (l-l') T_s\big).$$

Per layout, $\mathbf R_{\mathcal{PP}} = \mathbf U \boldsymbol\Lambda \mathbf U^H$ and $\mathbf A = \mathbf R_{\mathcal{GP}}\mathbf U$
are computed once; per slot, for any noise variance,

$$\hat{\mathbf h}_b = \mathbf A_b(\boldsymbol\Lambda_b + \sigma^2\mathbf I)^{-1}\mathbf U_b^H \hat{\mathbf h}_{\mathcal P}.$$

The bank uses $\tau_{\max} = 5.52\ \mu$s and $f_D \in \{100, 400, 1000\}$ Hz (`pilotformer.estimators.RobustFilterBank`).

## PilotFormer-R

1. **Pilot tokens.** $\mathbf x_p = \mathbf W_{\text{in}}[\Re \hat h_p, \Im \hat h_p]^T + \varphi(\gamma(\mathbf c_p)) + \mathbf e(\sigma^2)$,
   with Fourier features $\gamma$ of the normalized coordinates and an SNR embedding $\mathbf e$.
2. **Locality-biased attention.** Every head adds $-\alpha_h \Delta k^2 - \beta_h \Delta l^2$ to its logits, with learned
   $\alpha_h, \beta_h > 0$ (`LocalityBiasedAttention`).
3. **Set encoder.** 4 self-attention blocks over the pilot set.
4. **Grid decoder.** Every RE is a query built from its coordinates, a convolutional stem over the interpolated and bank
   estimates, and the SNR embedding; 3 cross-attention blocks attend to the encoded pilots.
5. **Head.** $\hat H(q) = \sum_b \operatorname{softmax}(\mathbf w_q)_b\,\hat h_b[q] + \Delta_q$, initialized to zero so that
   training starts from the average of the bank.

No layer has a weight indexed by the pilot index or sized by the number of pilots, so the model accepts any layout and is
invariant to the order of the pilots.

## Training

Each step draws one layout from the training family (pilot symbols every 3 to 7 symbols, pilot sub-carriers every 2, 3, 4
or 6, random offsets, stagger with probability 1/2: 750 layouts with 24 to 180 pilots) and a batch of fresh EPA/EVA
channels with delays stretched by $\mathcal U[0.5, 2.2]$, $f_D \sim \mathcal U[0, 1000]$ Hz and SNR $\sim \mathcal U[0, 30]$ dB.
The loss is the per-slot NMSE in the log domain. AdamW, one-cycle schedule with peak $5\times10^{-4}$, gradient clipping at
1, FP16, 20 000 steps of 128 slots.

## Evaluation

| Benchmark | Content |
|---|---|
| SNR sweep | 6 channels (EVA 200/750/1200 Hz, ETU, TDL-A, TDL-C) x 4 layouts x 0 to 30 dB, NMSE and 16-QAM BER |
| Doppler sweep | EVA, LTE-CRS, 20 dB, 0 to 1400 Hz (training covers 0 to 1000 Hz) |
| SNR mismatch | SNR given to SNR-aware estimators off by -3 to +3 dB |
| Complexity | parameters, MFLOP per slot, FP16 latency |

All estimators see the same channel realizations, noise and data bits.
