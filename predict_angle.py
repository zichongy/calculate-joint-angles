"""
Gaussian Process
"""
# import matplotlib.pyplot as plt
        # from sklearn.gaussian_process import GaussianProcessRegressor
        # from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared

        # kernel = 1.0 * RBF(length_scale=40.0, length_scale_bounds=(1e-1, 2e2))
        # # kernel = ExpSineSquared(length_scale=20.0, periodicity=1)
        # gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, random_state=0)

        # for key in filtered_kpts.keys():
        #     if 'elbow_angles' in key:
        #         X = np.linspace(start=0,stop=30,num=31)
        #         y = filtered_kpts[key][:,0]
        #         plt.plot(X, y, label=key)

        #         for i in range(10,30):
        #             X10 = X[0:i].reshape(-1,1)
        #             y10 = y[0:i]
        #             gpr.fit(X10, y10)
        #             gpr.kernel_
        #             mean_prediction, std_prediction = gpr.predict(X[i+1].reshape(-1, 1), return_std=True)
        #             plt.plot(X[i+1], mean_prediction, marker="x")
        # plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        # plt.show()

