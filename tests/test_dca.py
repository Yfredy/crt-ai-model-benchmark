import numpy as np

from analysis.dca import net_benefit, net_benefit_treat_all, net_benefit_treat_none

y = np.array([1, 1, 0, 0])
p = np.array([0.9, 0.8, 0.2, 0.1])
thr = np.array([0.5])

def test_perfect_case_threshold_05():
    # tp=2, fp=0, n=4 -> nb = 2/4 = 0.5
    assert np.isclose(net_benefit(y, p, thr)[0], 0.5)

def test_treat_all_formula():
    # treat-all: prevalence - (1-prevalence)*pt/(1-pt)；y=[1,1,0,0] 患病率 0.5
    assert np.isclose(net_benefit_treat_all(y, np.array([0.5]))[0], 0.0)
    # pt=0.25: 0.5 - 0.5*(0.25/0.75) = 1/3
    assert np.isclose(net_benefit_treat_all(y, np.array([0.25]))[0], 1 / 3)
    # pt=0.8: 0.5 - 0.5*0.8/0.2 = -1.5
    assert np.isclose(net_benefit_treat_all(y, np.array([0.8]))[0], -1.5)

def test_treat_all_consistent_with_net_benefit():
    # "全部预测为阳性"的模型净收益应等于 treat-all 曲线
    pp = np.ones(4)
    for pt in (0.1, 0.3, 0.6):
        assert np.isclose(net_benefit(y, pp, np.array([pt]))[0],
                          net_benefit_treat_all(y, np.array([pt]))[0])

def test_treat_none_zero():
    assert np.allclose(net_benefit_treat_none(np.array([0.1, 0.9])), 0.0)

def test_false_positive_penalty():
    # y=[1,0], p=[0.9,0.9], pt=0.5: tp=1, fp=1, n=2 -> 0.5 - 0.5 = 0
    yy = np.array([1, 0]); pp = np.array([0.9, 0.9])
    assert np.isclose(net_benefit(yy, pp, np.array([0.5]))[0], 0.0)

def test_nan_outside_valid_range():
    thr = np.array([0.0, 0.5, 1.0])
    out = net_benefit(y, p, thr)
    assert np.isnan(out[0]) and np.isnan(out[2]) and not np.isnan(out[1])

def test_matches_sklearn_style_example():
    # n=10, prevalence 0.4, pt=0.2, 全部预测阳性且 4 个真阳性:
    # nb = 4/10 - 6/10*(0.2/0.8) = 0.4 - 0.15 = 0.25
    yy = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pp = np.full(10, 0.9)
    assert np.isclose(net_benefit(yy, pp, np.array([0.2]))[0], 0.25)
