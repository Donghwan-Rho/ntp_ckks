# mean-1회 + thereafter ReLU-only + Lehmer-L normalization scheme

import os
os.environ["PYTHONHASHSEED"] = "42"

import random
random.seed(42)

import numpy as np
np.random.seed(42)

def cutmax(
    X: np.ndarray,
    p: int = 15,            # sharpening power (가능하면 홀수 권장)
    T: int = 6,             # iterations
    r: float = 1.0,         # Lehmer order: L_r = S_{r+1}/S_r (r>=0, 기본 1)
    eps: float = 1e-20,
    print_topk: int | None = 10,
    return_history: bool = False
):
    """
    Iter 1: a_plus = relu(Y - mean(Y))  -> L = S_{r+1}/(S_r+eps) on a_plus
            u = (Y - L) / (L + eps);  Y <- u**p
    Iter 2..T: a_plus = relu(Y) -> L = S_{r+1}/(S_r+eps) on a_plus
               u = (Y - L) / (L + eps);  Y <- u**p
    마지막에 L1-norm 정규화(Z = Y / sum(Y))
    """
    def _print_topk(label: str, V: np.ndarray, Vp: np.ndarray, X0: np.ndarray, k: int):
        k = min(k, V.size)
        idx_k = np.argpartition(V, -k)[-k:]
        idx_sorted = idx_k[np.argsort(-V[idx_k])]
        print(f"[{label}] Top {k} elements:")
        if label == "Final Z (normalized)":
            for rank, i in enumerate(idx_sorted, 1):
                print(f"  {rank:2d}) idx={i:6d}, Y={V[i]:.12g}, Z={Vp[i]:.12g}, X={X0[i]:.12g}")
        else:
            for rank, i in enumerate(idx_sorted, 1):
                print(f"  {rank:2d}) idx={i:6d}, u={V[i]:.12g}, u^p={Vp[i]:.12g}, X={X0[i]:.12g}")
        print("-" * 60)

    X = np.asarray(X, dtype=np.float64).reshape(-1)
    Y = X.copy()
    hist = [Y.copy()] if return_history else None

    for t in range(1, T + 1):
        # --- ReLU 단계 (iter=1만 mean 기준, 이후는 그냥 ReLU(Y)) ---
        if t == 1:
            tau = Y.mean()
            a_plus = np.maximum(Y - tau, 0.0)   # 공짜 ReLU 1회
            print(f"[Iter {t}] mean={tau:.6g}, pos={np.count_nonzero(a_plus>0)}, nonpos={a_plus.size-np.count_nonzero(a_plus>0)}")
        else:
            a_plus = np.maximum(Y, 0.0)         # 공짜 ReLU 1회
            print(f"[Iter {t}] pos={np.count_nonzero(a_plus>0)}, nonpos={a_plus.size-np.count_nonzero(a_plus>0)}")

        # --- Lehmer L_r = S_{r+1} / S_r on a_plus ---
        # S_r: sum (a_plus**r), S_{r+1}: sum (a_plus**(r+1))
        # r=0일 때 S_0는 양수 항의 개수로 해석
        if abs(r) < 1e-12:
            S_r = float((a_plus > 0).sum())
        else:
            S_r = float(np.sum(a_plus ** r))
        S_rp1 = float(np.sum(a_plus ** (r + 1.0)))

        # if S_r <= eps or S_rp1 <= 0.0:
        #     # 양의 초과량 거의 없음 → 소멸
        #     Y[:] = 0.0
        #     if isinstance(print_topk, int) and print_topk > 0:
        #         _print_topk(f"Iter {t}", Y, Y, X, print_topk)
        #     continue

        L = S_rp1 / (S_r + eps)   # Lehmer mean

        # --- Lehmer 중심 정규화 & 샤프닝 ---
        # u = (Y - L) / (L + eps)  (전역 스칼라 1회 나눗셈)
        invL = 1.0 / (L + eps)
        u = (Y - L) * invL
        print(f'before sharpening')
        Y = u ** p
        print(f'after sharpening')

        if return_history:
            hist.append(Y.copy())

        if isinstance(print_topk, int) and print_topk > 0:
            _print_topk(f"Iter {t}", u, Y, X, print_topk)

    # --- 마지막 정규화 (합 1) ---
    Y = np.maximum(Y, 0.0)
    S = Y.sum()
    Z = np.zeros_like(Y)
    if abs(S) >= eps:
        Z = Y / S

    if isinstance(print_topk, int) and print_topk > 0:
        _print_topk("Final Z (normalized)", Y, Z, X, print_topk)

    return (Z, hist) if return_history else Z


# ---------------------- 사용 예시 ----------------------
if __name__ == "__main__":
    z = np.random.uniform(-10, 10, 32000).astype(np.float64)
    # z[np.argmax(z)] += 0.001  # 선택 옵션

    # p는 홀수 권장(음수/양수 보존하여 다음 스텝 ReLU가 더 또렷이 컷)
    Z = cutmax(z, p=15, T=3, r=1.0, print_topk=5, return_history=False)

    approx_argmax = int(np.argmax(Z))
    true_argmax   = int(np.argmax(z))
    print(f"approx argmax = {approx_argmax}, true argmax = {true_argmax}")
