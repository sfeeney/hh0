import matplotlib.pyplot as mp
mp.rc("font", family="serif", size=12)
mp.rc("text", usetex=True)

import daft

# @TODO: increase font?
# @TODO: calibration in all Cepheids; comes into measurement...
# @TODO: full version including SNe complexity[, heavy tails?]
#        uncertain periods and metals? makes things way more complex
#        and not sure we know enough about how it works: p & m correlated?

s_color = {"ec": "#f89406"}
p_color = {"ec": "#46a546"}

fix_redshifts = False

if fix_redshifts:
	pgm_smf = daft.PGM([19.5, 9.0], origin=[0.3, 0.6], observed_style="inner")
	pgm_full = daft.PGM([19.5, 9.0], origin=[0.3, 0.6], observed_style="inner")
	pgm_r16 = daft.PGM([15.75, 9.0], origin=[0.3, 0.6], observed_style="inner")
else:
	pgm_smf = daft.PGM([20.5, 9.0], origin=[0.3, 0.6], observed_style="inner")
	pgm_full = daft.PGM([20.5, 9.0], origin=[0.3, 0.6], observed_style="inner")
	pgm_r16 = daft.PGM([15.75, 9.0], origin=[0.3, 0.6], observed_style="inner")
	pgm_ms = daft.PGM([4.4, 3.8], origin=[0.3, 0.6], observed_style="inner")

# quick cutout of the model-selection portion
'''
lmarge = 2
hspace = 1.5
pgm_ms.add_node(daft.Node("ph0", r"${\rm Pr}(H_0)$", lmarge, 9, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("pq0", r"${\rm Pr}(q_0)$", lmarge + hspace, 9, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("pdh0", r"${\rm Pr}(\Delta H_0)$", lmarge + 2 * hspace, 9, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("pdq0", r"${\rm Pr}(\Delta q_0)$", lmarge + 3 * hspace, 9, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("h0", r"$H_0$", lmarge, 8))
pgm_ms.add_node(daft.Node("q0", r"$q_0$", lmarge + hspace, 8))
pgm_ms.add_node(daft.Node("dh0", r"$\Delta H_0$", lmarge + 2 * hspace, 8))
pgm_ms.add_node(daft.Node("dq0", r"$\Delta q_0$", lmarge + 3 * hspace, 8))
pgm_ms.add_node(daft.Node("ins", r" ", 0.3, 7.75, scale=0.0))
pgm_ms.add_edge("ph0", "h0")
pgm_ms.add_edge("pdh0", "dh0")
pgm_ms.add_edge("pq0", "q0")
pgm_ms.add_edge("pdq0", "dq0")
pgm_ms.add_node(daft.Node("pmsk", r"${\rm Pr}(m_i^{\rm s}|z_i^{\rm s},x^{\rm s}_i,c^{\rm s}_i,M^{\rm s},\alpha^{\rm s},\beta^{\rm s},H_0,q_0,\sigma^{\rm int,\,s})$", \
						  2.2, 7, aspect = 7.2, plot_params=s_color, shape="rectangle"))
pgm_ms.add_node(daft.Node("phqchat", r"${\rm Pr}(\hat{H}_0,\hat{q}_0|H_0+\Delta H_0,q_0+\Delta q_0,\Sigma_{\rm cos})$", \
						  lmarge + 2.5 * hspace, 7, aspect = 6.6, plot_params=s_color, shape="rectangle"))
in_edge = pgm_ms.add_edge("ins", "pmsk")
pgm_ms.add_edge("h0", "pmsk")
pgm_ms.add_edge("q0", "pmsk")
pgm_ms.add_edge("h0", "phqchat")
pgm_ms.add_edge("q0", "phqchat")
pgm_ms.add_edge("dh0", "phqchat")
pgm_ms.add_edge("dq0", "phqchat")
pgm_ms.add_node(daft.Node("h0chat", r"$\hat{H}_0$", lmarge + 2 * hspace, 6, observed=True))
pgm_ms.add_node(daft.Node("q0chat", r"$\hat{q}_0$", lmarge + 3 * hspace, 6, observed=True))
pgm_ms.add_node(daft.Node("outs", r" ", 0.3, 6.25, scale=0.0))
pgm_ms.add_edge("phqchat", "h0chat")
pgm_ms.add_edge("phqchat", "q0chat")
out_edge = pgm_ms.add_edge("pmsk", "outs")
in_edge.plot_params = {'linestyle': '--'}
out_edge.plot_params = {'linestyle': '--'}
pgm_ms.render()
pgm_ms.figure.savefig("bhm_model_selection_extension.pdf")
'''
lmarge = 1
hspace = 1
bmarge = 0
vspace = 1
pgm_ms.add_node(daft.Node("ph0", r"${\rm Pr}(H_0)$", lmarge, bmarge + 4 * vspace, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("pq0", r"${\rm Pr}(q_0)$", lmarge + hspace, bmarge + 4 * vspace, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("pdh0", r"${\rm Pr}(\Delta H_0)$", lmarge + 2 * hspace, bmarge + 4 * vspace, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("pdq0", r"${\rm Pr}(\Delta q_0)$", lmarge + 3 * hspace, bmarge + 4 * vspace, \
	                      aspect = 1.8, plot_params=s_color, \
	                      shape="rectangle"))
pgm_ms.add_node(daft.Node("h0", r"$H_0$", lmarge, bmarge + 3 * vspace))
pgm_ms.add_node(daft.Node("q0", r"$q_0$", lmarge + hspace, bmarge + 3 * vspace))
pgm_ms.add_node(daft.Node("dh0", r"$\Delta H_0$", lmarge + 2 * hspace, bmarge + 3 * vspace))
pgm_ms.add_node(daft.Node("dq0", r"$\Delta q_0$", lmarge + 3 * hspace, bmarge + 3 * vspace))
pgm_ms.add_edge("ph0", "h0")
pgm_ms.add_edge("pdh0", "dh0")
pgm_ms.add_edge("pq0", "q0")
pgm_ms.add_edge("pdq0", "dq0")
pgm_ms.add_node(daft.Node("outh", r" ", 0, bmarge + 2 * vspace + 0.30, scale=0.0))
pgm_ms.add_node(daft.Node("outq", r" ", 0, bmarge + 2 * vspace + 0.25, scale=0.0))
pgm_ms.add_node(daft.Node("phqchat", r"${\rm Pr}(\hat{H}_0,\hat{q}_0|H_0+\Delta H_0,q_0+\Delta q_0,\Sigma_{\rm cos})$", \
						  lmarge + 1.5 * hspace, bmarge + 2 * vspace, aspect = 6.6, plot_params=s_color, shape="rectangle"))
out_edge_h = pgm_ms.add_edge("h0", "outh", directed=False)
out_edge_q = pgm_ms.add_edge("q0", "outq", directed=False)
pgm_ms.add_edge("h0", "phqchat")
pgm_ms.add_edge("q0", "phqchat")
pgm_ms.add_edge("dh0", "phqchat")
pgm_ms.add_edge("dq0", "phqchat")
pgm_ms.add_node(daft.Node("h0chat", r"$\hat{H}_0$", \
	                      lmarge + hspace, bmarge + vspace, \
	                      observed=True))
pgm_ms.add_node(daft.Node("q0chat", r"$\hat{q}_0$", \
	                      lmarge + 2 * hspace, bmarge + vspace, \
	                      observed=True))
pgm_ms.add_edge("phqchat", "h0chat")
pgm_ms.add_edge("phqchat", "q0chat")
out_edge_h.plot_params = {'linestyle': '--'}
out_edge_q.plot_params = {'linestyle': '--'}
pgm_ms.render()
pgm_ms.figure.savefig("bhm_model_selection_extension.pdf")
exit()

# common nodes
for pgm in [pgm_smf, pgm_full, pgm_r16]:

	# Prob dists.
	pgm.add_node(daft.Node("pmu", r"${\rm Pr}(\mu)$", 1, 9, \
		                   aspect = 1.1, plot_params=s_color, \
		                   shape="rectangle"))
	pgm.add_node(daft.Node("pmubranch", r" ", 1, 7, scale=0.0))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("psp", r"${\rm Pr}(s^p)$", 4, 9, \
			                   aspect = 1.2, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("psZ", r"${\rm Pr}(s^Z)$", 5, 9, \
			                   aspect = 1.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pp", r"${\rm Pr}(p^{\rm c})$", 6, 9, \
			                   aspect = 1.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pZ", r"${\rm Pr}(Z^{\rm c})$", 7, 9, \
			                   aspect = 1.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmc", r"${\rm Pr}(M^{\rm c})$", 8, 9, \
			                   aspect = 1.7, plot_params=s_color, \
			                   shape="rectangle"))
	else:	
		pgm.add_node(daft.Node("psp", r"${\rm Pr}(s^p)$", 5, 9, \
			                   aspect = 1.2, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("psZ", r"${\rm Pr}(s^Z)$", 6, 9, \
			                   aspect = 1.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmc", r"${\rm Pr}(M^{\rm c})$", 7, 9, \
			                   aspect = 1.7, plot_params=s_color, \
			                   shape="rectangle"))
	if pgm == pgm_r16:
		pgm.add_node(daft.Node("pms", r"${\rm Pr}(M^{\rm s})$", 12.25, 9, \
			                   aspect = 1.7, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pax", r"${\rm Pr}(a_x)$", 15.25, 9, \
			                   aspect = 1.3, plot_params=s_color, \
			                   shape="rectangle"))
	else:
		if pgm == pgm_full:
			pgm.add_node(daft.Node("psigintc", r"${\rm Pr}(\sigma^{\rm int,\,c})$", 9, \
				                   9, aspect = 1.7, plot_params=s_color, \
				                   shape="rectangle"))
		else:
			pgm.add_node(daft.Node("psigintc", r"${\rm Pr}(\sigma^{\rm int,\,c})$", 8, \
				                   9, aspect = 1.7, plot_params=s_color, \
				                   shape="rectangle"))
		pgm.add_node(daft.Node("pms", r"${\rm Pr}(M^{\rm s})$", 12.25, 9, \
			                   aspect = 1.7, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("psigints", r"${\rm Pr}(\sigma^{\rm int,\,s})$", \
			                   13.25, 9, aspect = 1.7, \
			                   plot_params=s_color, shape="rectangle"))
		if pgm == pgm_full:
			pgm.add_node(daft.Node("pxs", r"${\rm Pr}(x^{\rm s})$", 14.22, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("pcs", r"${\rm Pr}(c^{\rm s})$", 15.155, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("pas", r"${\rm Pr}(\alpha^{\rm s})$", 16.125, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("pbs", r"${\rm Pr}(\beta^{\rm s})$", 17.125, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("ph0", r"${\rm Pr}(H_0)$", 18.125, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			if fix_redshifts:
				pgm.add_node(daft.Node("pq0", r"${\rm Pr}(q_0)$", 19, 9, \
					                   aspect = 1.2, plot_params=s_color, \
				    	               shape="rectangle"))
			else:
				pgm.add_node(daft.Node("pz", r"${\rm Pr}(z^{\rm s})$", 19.125, 9, \
					                   aspect = 1.2, plot_params=s_color, \
				    	               shape="rectangle"))
				pgm.add_node(daft.Node("pq0", r"${\rm Pr}(q_0)$", 20.125, 9, \
					                   aspect = 1.2, plot_params=s_color, \
				    	               shape="rectangle"))
		else:
			pgm.add_node(daft.Node("pxs", r"${\rm Pr}(x^{\rm s})$", 14.22, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("pcs", r"${\rm Pr}(c^{\rm s})$", 15.03, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("pas", r"${\rm Pr}(\alpha^{\rm s})$", 15.925, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("pbs", r"${\rm Pr}(\beta^{\rm s})$", 16.925, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			pgm.add_node(daft.Node("ph0", r"${\rm Pr}(H_0)$", 18.925, 9, \
				                   aspect = 1.4, plot_params=s_color, \
				                   shape="rectangle"))
			if fix_redshifts:
				pgm.add_node(daft.Node("pq0", r"${\rm Pr}(q_0)$", 19, 9, \
					                   aspect = 1.2, plot_params=s_color, \
				    	               shape="rectangle"))
			else:
				pgm.add_node(daft.Node("pz", r"${\rm Pr}(z^{\rm s})$", 17.925, 9, \
					                   aspect = 1.2, plot_params=s_color, \
				    	               shape="rectangle"))
				pgm.add_node(daft.Node("pq0", r"${\rm Pr}(q_0)$", 19.925, 9, \
					                   aspect = 1.2, plot_params=s_color, \
				    	               shape="rectangle"))

	# Hierarchical parameters.
	if pgm == pgm_full:
		pgm.add_node(daft.Node("sp", r"$s^p$", 4, 8))
		pgm.add_node(daft.Node("sZ", r"$s^Z$", 5, 8))
		pgm.add_node(daft.Node("mc", r"$M^{\rm c}$", 8, 8))
	else:
		pgm.add_node(daft.Node("sp", r"$s^p$", 5, 8))
		pgm.add_node(daft.Node("sZ", r"$s^Z$", 6, 8))
		pgm.add_node(daft.Node("mc", r"$M^{\rm c}$", 7, 8))
	if pgm == pgm_r16:
		pgm.add_node(daft.Node("sigintc", r"$\sigma^{\rm int,\,c}$", 8, 8, fixed = True))
		pgm.add_node(daft.Node("sigints", r"$\sigma^{\rm int,\,s}$", 11.25, 8, fixed = True))
		pgm.add_node(daft.Node("ms", r"$M^{\rm s}$", 12.25, 8))
		pgm.add_node(daft.Node("ax", r"$a_x$", 15.25, 8))
		pgm.add_node(daft.Node("h0", r"$H_0$", 13.75, 8, fixed = True))
	else:
		if pgm == pgm_full:
			pgm.add_node(daft.Node("sigintc", r"$\sigma^{\rm int,\,c}$", 9, 8))
		else:
			pgm.add_node(daft.Node("sigintc", r"$\sigma^{\rm int,\,c}$", 8, 8))
		pgm.add_node(daft.Node("ms", r"$M^{\rm s}$", 12.25, 8))
		pgm.add_node(daft.Node("sigints", r"$\sigma^{\rm int,\,s}$", 13.25, 8))
		if pgm == pgm_full:
			pgm.add_node(daft.Node("as", r"$\alpha^{\rm s}$", 16.125, 8))
			pgm.add_node(daft.Node("bs", r"$\beta^{\rm s}$", 17.125, 8))
			pgm.add_node(daft.Node("h0", r"$H_0$", 18.125, 8))
			if fix_redshifts:
				pgm.add_node(daft.Node("q0", r"$q_0$", 19, 8))
			else:
				pgm.add_node(daft.Node("q0", r"$q_0$", 20.125, 8))
		else:
			pgm.add_node(daft.Node("as", r"$\alpha^{\rm s}$", 15.925, 8))
			pgm.add_node(daft.Node("bs", r"$\beta^{\rm s}$", 16.925, 8))
			pgm.add_node(daft.Node("h0", r"$H_0$", 18.925, 8))
			if fix_redshifts:
				pgm.add_node(daft.Node("q0", r"$q_0$", 19, 8))
			else:
				pgm.add_node(daft.Node("q0", r"$q_0$", 19.925, 8))

	# Edges.
	pgm.add_edge("psp", "sp")
	pgm.add_edge("psZ", "sZ")
	pgm.add_edge("pmc", "mc")
	if pgm == pgm_r16:
		pgm.add_edge("pms", "ms")
		pgm.add_edge("pax", "ax")
		pgm.add_edge("h0", "ms")
		pgm.add_edge("h0", "ax")
	else:
		pgm.add_edge("psigintc", "sigintc")
		pgm.add_edge("pms", "ms")
		pgm.add_edge("psigints", "sigints")
		pgm.add_edge("pas", "as")
		pgm.add_edge("pbs", "bs")
		pgm.add_edge("ph0", "h0")
		pgm.add_edge("pq0", "q0")

	# Latent variable.
	pgm.add_node(daft.Node("mu0", r"$\mu_i$", 1, 6))
	pgm.add_node(daft.Node("mui", r"$\mu_i$", 10, 6))
	pgm.add_node(daft.Node("pmubend", r" ", 10, 7, scale=0.0))
	if pgm == pgm_smf:
		pgm.add_node(daft.Node("xskbend1", r" ", 14.22, 7, scale=0.0))
		pgm.add_node(daft.Node("cskbend1", r" ", 15.03, 6.95, scale=0.0))
		pgm.add_node(daft.Node("xskbend2", r" ", 13.925, 7, scale=0.0))
		pgm.add_node(daft.Node("cskbend2", r" ", 13.875, 6.95, scale=0.0))
		pgm.add_node(daft.Node("xskbend3", r" ", 13.925, 4.25, scale=0.0))
		pgm.add_node(daft.Node("cskbend3", r" ", 13.875, 3.75, scale=0.0))
	elif pgm == pgm_full:
		pgm.add_node(daft.Node("pbend1", r" ", 6, 8, scale=0.0))
		pgm.add_node(daft.Node("Zbend1", r" ", 7, 8, scale=0.0))
		pgm.add_node(daft.Node("pbend2", r" ", 6.4275, 8, scale=0.0))
		pgm.add_node(daft.Node("Zbend2", r" ", 6.5725, 8, scale=0.0))
		pgm.add_node(daft.Node("pbend3", r" ", 6.4275, 4.175, scale=0.0))
		pgm.add_node(daft.Node("Zbend3", r" ", 6.5725, 3.825, scale=0.0))
		pgm.add_node(daft.Node("xskbend1", r" ", 14.22, 8, scale=0.0))
		pgm.add_node(daft.Node("cskbend1", r" ", 15.155, 8, scale=0.0))
		pgm.add_node(daft.Node("xskbend2", r" ", 14.615, 8, scale=0.0))
		pgm.add_node(daft.Node("cskbend2", r" ", 14.760, 8, scale=0.0))
		pgm.add_node(daft.Node("xskbend3", r" ", 14.615, 4.25, scale=0.0))
		pgm.add_node(daft.Node("cskbend3", r" ", 14.760, 3.75, scale=0.0))

	# Edges.
	pgm.add_edge("pmu", "mu0")
	pgm.add_edge("pmubranch", "pmubend", directed=False)
	pgm.add_edge("pmubend", "mui")
	if pgm != pgm_r16:
		if pgm == pgm_full:
			pgm.add_edge("pp", "pbend1", directed=False)
			pgm.add_edge("pZ", "Zbend1", directed=False)
			pgm.add_edge("pbend1", "pbend2", directed=False)
			pgm.add_edge("Zbend1", "Zbend2", directed=False)
			pgm.add_edge("pbend2", "pbend3", directed=False)
			pgm.add_edge("Zbend2", "Zbend3", directed=False)
		pgm.add_edge("pxs", "xskbend1", directed=False)
		pgm.add_edge("pcs", "cskbend1", directed=False)
		pgm.add_edge("xskbend1", "xskbend2", directed=False)
		pgm.add_edge("cskbend1", "cskbend2", directed=False)
		pgm.add_edge("xskbend2", "xskbend3", directed=False)
		pgm.add_edge("cskbend2", "cskbend3", directed=False)

	# Prob dists.
	pgm.add_node(daft.Node("pm0j", r"${\rm Pr}(m_{ij}^{\rm c}|\mu_i,\hat{p}_{ij}^{\rm c},\hat{Z}_{ij}^{\rm c},s^p,s^Z,M^{\rm c},\sigma^{\rm int,\,c})$", \
		                   4, 5, \
		                   aspect = 6.7, plot_params=s_color, \
		                   shape="rectangle"))
	pgm.add_node(daft.Node("pmij", r"${\rm Pr}(m_{ij}^{\rm c}|\mu_i,\hat{p}_{ij}^{\rm c},\hat{Z}_{ij}^{\rm c},s^p,s^Z,M^{\rm c},\sigma^{\rm int,\,c})$", 9, 5, \
		                   aspect = 6.5, plot_params=s_color, \
		                   shape="rectangle"))
	pgm.add_node(daft.Node("pmsi", r"${\rm Pr}(m_i^{\rm s}|\mu_i,M^{\rm s},\sigma^{\rm int,\,s})$", 12.25, 5, \
		                   aspect = 3.6, plot_params=s_color, \
		                   shape="rectangle"))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("pmsi", r"${\rm Pr}(m_i^{\rm s}|\mu_i,x^{\rm s}_i,c^{\rm s}_i,M^{\rm s},\alpha^{\rm s},\beta^{\rm s},\sigma^{\rm int,\,s})$", 12.75, 5, \
			                   aspect = 6.0, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmsk", r"${\rm Pr}(m_i^{\rm s}|z_i^{\rm s},x^{\rm s}_i,c^{\rm s}_i,M^{\rm s},\alpha^{\rm s},\beta^{\rm s},H_0,q_0,\sigma^{\rm int,\,s})$", 17.125, 5, \
			                   aspect = 7.2, plot_params=s_color, \
			                   shape="rectangle"))
	elif pgm == pgm_r16:
		pgm.add_node(daft.Node("pmsi", r"${\rm Pr}(m_i^{\rm s}|\mu_i,M^{\rm s},\sigma^{\rm int,\,s})$", 12.25, 5, \
			                   aspect = 3.6, plot_params=s_color, \
			                   shape="rectangle"))
	else:
		pgm.add_node(daft.Node("pmsk", r"${\rm Pr}(m_i^{\rm s}|z_i^{\rm s},x^{\rm s}_i,c^{\rm s}_i,M^{\rm s},\alpha^{\rm s},\beta^{\rm s},H_0,q_0,\sigma^{\rm int,\,s})$", 15.925, 5, \
			                   aspect = 7.2, plot_params=s_color, \
			                   shape="rectangle"))

	# Edges.
	pgm.add_edge("mu0", "pm0j")
	pgm.add_edge("sp", "pm0j")
	pgm.add_edge("sZ", "pm0j")
	pgm.add_edge("mc", "pm0j")
	pgm.add_edge("sigintc", "pm0j")
	pgm.add_edge("mui", "pmij")
	pgm.add_edge("sp", "pmij")
	pgm.add_edge("sZ", "pmij")
	pgm.add_edge("mc", "pmij")
	pgm.add_edge("sigintc", "pmij")
	pgm.add_edge("mui", "pmsi")
	pgm.add_edge("ms", "pmsi")
	pgm.add_edge("sigints", "pmsi")
	if pgm != pgm_r16:
		pgm.add_edge("ms", "pmsk")
		pgm.add_edge("sigints", "pmsk")
		pgm.add_edge("as", "pmsk")
		pgm.add_edge("bs", "pmsk")
		pgm.add_edge("h0", "pmsk")
		pgm.add_edge("q0", "pmsk")
		if pgm == pgm_full:
			pgm.add_edge("as", "pmsi")
			pgm.add_edge("bs", "pmsi")


	# Latent variable.
	if pgm == pgm_full:
		pgm.add_node(daft.Node("m0j", r"$m_{ij}^{\rm c}$", 3, 3.825))
		pgm.add_node(daft.Node("p0j", r"$p_{ij}^{\rm c}$", 4, 4.175))
		pgm.add_node(daft.Node("Z0j", r"$Z_{ij}^{\rm c}$", 5, 3.825))
	else:
		pgm.add_node(daft.Node("m0j", r"$m_{ij}^{\rm c}$", 3, 4))
		pgm.add_node(daft.Node("Zhat0bend", r" ", 5, 4.1, scale=0.0))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("mij", r"$m_{ij}^{\rm c}$", 10, 3.825))
		pgm.add_node(daft.Node("pij", r"$p_{ij}^{\rm c}$", 9, 4.175))
		pgm.add_node(daft.Node("Zij", r"$Z_{ij}^{\rm c}$", 8, 3.825))
	else:
		pgm.add_node(daft.Node("mij", r"$m_{ij}^{\rm c}$", 8, 4))
		pgm.add_node(daft.Node("Zhatibend", r" ", 10, 4.1, scale=0.0))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("msi", r"$m_i^{\rm s}$", 12.75, 4))
	else:
		pgm.add_node(daft.Node("msi", r"$m_i^{\rm s}$", 12.25, 4))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("xsi", r"$x_i^{\rm s}$", 13.75, 4.25))
		pgm.add_node(daft.Node("csi", r"$c_i^{\rm s}$", 13.75, 3.75))
		pgm.add_node(daft.Node("msk", r"$m_i^{\rm s}$", 16.625, 4))
		pgm.add_node(daft.Node("xsk", r"$x_i^{\rm s}$", 15.625, 4.25))
		pgm.add_node(daft.Node("csk", r"$c_i^{\rm s}$", 15.625, 3.75))
		if fix_redshifts:
			pgm.add_node(daft.Node("zhatskbend", r" ", 18.625, 4.1, scale=0.0))
		else:
			pgm.add_node(daft.Node("zsk", r"$z_i^{\rm s}$", 18.625, 4))
			pgm.add_node(daft.Node("zhatskbend", r" ", 19.125, 4, scale=0.0))
	elif pgm == pgm_smf:
		pgm.add_node(daft.Node("msk", r"$m_i^{\rm s}$", 15.425, 4))
		pgm.add_node(daft.Node("xsk", r"$x_i^{\rm s}$", 14.425, 4.25))
		pgm.add_node(daft.Node("csk", r"$c_i^{\rm s}$", 14.425, 3.75))
		if fix_redshifts:
			pgm.add_node(daft.Node("zhatskbend", r" ", 17.425, 4.1, scale=0.0))
		else:
			pgm.add_node(daft.Node("zsk", r"$z_i^{\rm s}$", 17.425, 4))
			pgm.add_node(daft.Node("zhatskbend", r" ", 17.925, 4, scale=0.0))

	# Edges.
	pgm.add_edge("pm0j", "m0j")
	if pgm == pgm_full:
		pgm.add_edge("p0j", "pm0j")
		pgm.add_edge("Z0j", "pm0j")
	else:
		pgm.add_edge("Zhat0bend", "pm0j")
	pgm.add_edge("pmij", "mij")
	if pgm == pgm_full:
		pgm.add_edge("pij", "pmij")
		pgm.add_edge("Zij", "pmij")
	else:
		pgm.add_edge("Zhatibend", "pmij")
	pgm.add_edge("pmsi", "msi")
	if pgm != pgm_r16:
		if pgm == pgm_full:
			pgm.add_edge("pbend3", "p0j")
			pgm.add_edge("Zbend3", "Z0j")
			pgm.add_edge("pbend3", "pij")
			pgm.add_edge("Zbend3", "Zij")
			pgm.add_edge("xsi", "pmsi")
			pgm.add_edge("pmsi", "msi")
			pgm.add_edge("csi", "pmsi")
			pgm.add_edge("xskbend3", "xsi")
			pgm.add_edge("cskbend3", "csi")
		pgm.add_edge("xsk", "pmsk")
		pgm.add_edge("pmsk", "msk")
		pgm.add_edge("csk", "pmsk")
		pgm.add_edge("xskbend3", "xsk")
		pgm.add_edge("cskbend3", "csk")
		if not fix_redshifts:
			pgm.add_edge("pz", "zhatskbend", directed=False)
			pgm.add_edge("zhatskbend", "zsk")
			pgm.add_edge("zsk", "pmsk")

	# Prob dists.
	if pgm == pgm_r16:
		pgm.add_node(daft.Node("pdhat0", r"${\rm Pr}(\hat{\mu}_i|\mu_i,\sigma_{\mu_i})$", 1, 3, \
			                   aspect = 2.5, plot_params=s_color, \
			                   shape="rectangle"))
	else:
		pgm.add_node(daft.Node("pdhat0", r"${\rm Pr}(\hat{d}_i|\mu_i,\sigma_{d_i})$", 1, 3, \
			                   aspect = 2.5, plot_params=s_color, \
			                   shape="rectangle"))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("pmhat0", r"${\rm Pr}(\hat{m}_{ij}^{\rm c}|m_{ij}^{\rm c},\sigma_{m_{ij}^{\rm c}})$", 3, 2.7, \
			                   aspect = 3.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pphat0", r"${\rm Pr}(\hat{p}_{ij}^{\rm c}|p_{ij}^{\rm c},\sigma_{p_{ij}^{\rm c}})$", 4, 3.3, \
			                   aspect = 3.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pZhat0", r"${\rm Pr}(\hat{Z}_{ij}^{\rm c}|Z_{ij}^{\rm c},\sigma_{Z_{ij}^{\rm c}})$", 5, 2.7, \
			                   aspect = 3.2, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmhati", r"${\rm Pr}(\hat{m}_{ij}^{\rm c}|m_{ij}^{\rm c},\sigma_{m_{ij}^{\rm c}})$", 10, 2.7, \
			                   aspect = 3.2, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pphati", r"${\rm Pr}(\hat{p}_{ij}^{\rm c}|p_{ij}^{\rm c},\sigma_{p_{ij}^{\rm c}})$", 9, 3.3, \
			                   aspect = 3.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pZhati", r"${\rm Pr}(\hat{Z}_{ij}^{\rm c}|Z_{ij}^{\rm c},\sigma_{Z_{ij}^{\rm c}})$", 8, 2.7, \
			                   aspect = 3.2, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmxchatsi", r"${\rm Pr}(\hat{m}_i^{\rm s},\hat{x}_i^{\rm s},\hat{c}_i^{\rm s}|m_i^{\rm s},x_i^{\rm s},c_i^{\rm s},\Sigma_i)$", 12.75, 3, \
			                   aspect = 4.8, plot_params=s_color, \
			                   shape="rectangle"))
	else:
		pgm.add_node(daft.Node("pmhat0", r"${\rm Pr}(\hat{m}_{ij}^{\rm c}|m_{ij}^{\rm c},\sigma_{m_{ij}^{\rm c}})$", 3, 3, \
			                   aspect = 3.3, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmhati", r"${\rm Pr}(\hat{m}_{ij}^{\rm c}|m_{ij}^{\rm c},\sigma_{m_{ij}^{\rm c}})$", 8, 3, \
			                   aspect = 3.2, plot_params=s_color, \
			                   shape="rectangle"))
		pgm.add_node(daft.Node("pmhatsi", r"${\rm Pr}(\hat{m}_i^{\rm s}|m_i^{\rm s},\sigma_{m_i^{\rm s}})$", 12.25, 3, \
			                   aspect = 2.9, plot_params=s_color, \
			                   shape="rectangle"))
	if pgm == pgm_r16:
		pgm.add_node(daft.Node("paxhat", r"${\rm Pr}(\hat{a}_x|a_x,\sigma_{a_x})$", 15.25, 3, \
			                   aspect = 2.6, plot_params=s_color, \
		    	               shape="rectangle"))
	else:
		if pgm == pgm_full:
			pgm.add_node(daft.Node("pmxchatsk", r"${\rm Pr}(\hat{m}_i^{\rm s},\hat{x}_i^{\rm s},\hat{c}_i^{\rm s}|m_i^{\rm s},x_i^{\rm s},c_i^{\rm s},\Sigma_i)$", 16.625, 3, \
				                   aspect = 4.8, plot_params=s_color, \
				                   shape="rectangle"))
		else:
			pgm.add_node(daft.Node("pmxchatsk", r"${\rm Pr}(\hat{m}_i^{\rm s},\hat{x}_i^{\rm s},\hat{c}_i^{\rm s}|m_i^{\rm s},x_i^{\rm s},c_i^{\rm s},\Sigma_i)$", 15.425, 3, \
				                   aspect = 4.8, plot_params=s_color, \
				                   shape="rectangle"))
		if fix_redshifts:
			pgm.add_node(daft.Node("pq0hat", r"${\rm Pr}(\hat{q}_0|q_0,\sigma_{q_0})$", 19, 3, \
				                   aspect = 2.5, plot_params=s_color, \
			    	               shape="rectangle"))
		else:
			if pgm == pgm_full:
				pgm.add_node(daft.Node("pzhatsk", r"${\rm Pr}(\hat{z}_i^{\rm s}|z_i^{\rm s},\sigma_{z_i^{\rm s}})$", 18.625, 3, \
					                   aspect = 2.4, plot_params=s_color, \
				    	               shape="rectangle"))
				pgm.add_node(daft.Node("pq0hat", r"${\rm Pr}(\hat{q}_0|q_0,\sigma_{q_0})$", 20.125, 3, \
					                   aspect = 2.5, plot_params=s_color, \
				    	               shape="rectangle"))
			else:
				pgm.add_node(daft.Node("pzhatsk", r"${\rm Pr}(\hat{z}_i^{\rm s}|z_i^{\rm s},\sigma_{z_i^{\rm s}})$", 17.425, 3, \
					                   aspect = 2.4, plot_params=s_color, \
				    	               shape="rectangle"))
				pgm.add_node(daft.Node("pq0hat", r"${\rm Pr}(\hat{q}_0|q_0,\sigma_{q_0})$", 19.925, 3, \
					                   aspect = 2.5, plot_params=s_color, \
				    	               shape="rectangle"))

	# Edges.
	pgm.add_edge("mu0", "pdhat0")
	pgm.add_edge("m0j", "pmhat0")
	pgm.add_edge("mij", "pmhati")
	if pgm == pgm_r16:
		pgm.add_edge("msi", "pmhatsi")
		pgm.add_edge("ax", "paxhat")
	else:
		if pgm == pgm_full:
			pgm.add_edge("p0j", "pphat0")
			pgm.add_edge("Z0j", "pZhat0")
			pgm.add_edge("pij", "pphati")
			pgm.add_edge("Zij", "pZhati")
			pgm.add_edge("msi", "pmxchatsi")
			pgm.add_edge("xsi", "pmxchatsi")
			pgm.add_edge("msi", "pmxchatsi")
			pgm.add_edge("csi", "pmxchatsi")
		else:
			pgm.add_edge("msi", "pmhatsi")
		pgm.add_edge("xsk", "pmxchatsk")
		pgm.add_edge("msk", "pmxchatsk")
		pgm.add_edge("csk", "pmxchatsk")
		if not fix_redshifts:
			pgm.add_edge("zsk", "pzhatsk")
		pgm.add_edge("q0", "pq0hat")

	# Data.
	if pgm == pgm_r16:
		pgm.add_node(daft.Node("dhat0", r"$\hat{\mu}_i$", 1, 2, observed=True))
	else:
		pgm.add_node(daft.Node("dhat0", r"$\hat{d}_i$", 1, 2, observed=True))
	pgm.add_node(daft.Node("mhat0", r"$\hat{m}_{ij}^{\rm c}$", 3, 2, observed=True))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("phat0", r"$\hat{p}_{ij}^{\rm c}$", 4, 2, observed=True))
		pgm.add_node(daft.Node("Zhat0", r"$\hat{Z}_{ij}^{\rm c}$", 5, 2, observed=True))
	else:
		pgm.add_node(daft.Node("phat0", r"$\hat{p}_{ij}^{\rm c}$", 4, 2, observed=True, plot_params=p_color))
		pgm.add_node(daft.Node("Zhat0", r"$\hat{Z}_{ij}^{\rm c}$", 5, 2, observed=True, plot_params=p_color))
	if pgm == pgm_full:
		pgm.add_node(daft.Node("mhati", r"$\hat{m}_{ij}^{\rm c}$", 10, 2, observed=True))
		pgm.add_node(daft.Node("phati", r"$\hat{p}_{ij}^{\rm c}$", 9, 2, observed=True))
		pgm.add_node(daft.Node("Zhati", r"$\hat{Z}_{ij}^{\rm c}$", 8, 2, observed=True))
	else:
		pgm.add_node(daft.Node("mhati", r"$\hat{m}_{ij}^{\rm c}$", 8, 2, observed=True))
		pgm.add_node(daft.Node("phati", r"$\hat{p}_{ij}^{\rm c}$", 9, 2, observed=True, plot_params=p_color))
		pgm.add_node(daft.Node("Zhati", r"$\hat{Z}_{ij}^{\rm c}$", 10, 2, observed=True, plot_params=p_color))
	if pgm == pgm_r16:
		pgm.add_node(daft.Node("mhatsi", r"$\hat{m}_i^{\rm s}$", 12.25, 2, observed=True))
		pgm.add_node(daft.Node("axhat", r"$\hat{a}_x$", 15.25, 2, observed=True))
	else:
		if pgm == pgm_full:
			pgm.add_node(daft.Node("mhatsi", r"$\hat{m}_i^{\rm s}$", 11.75, 2, observed=True))
			pgm.add_node(daft.Node("xhatsi", r"$\hat{x}_i^{\rm s}$", 12.75, 2, observed=True))
			pgm.add_node(daft.Node("chatsi", r"$\hat{c}_i^{\rm s}$", 13.75, 2, observed=True))
			pgm.add_node(daft.Node("mhatsk", r"$\hat{m}_i^{\rm s}$", 15.625, 2, observed=True))
			pgm.add_node(daft.Node("xhatsk", r"$\hat{x}_i^{\rm s}$", 16.625, 2, observed=True))
			pgm.add_node(daft.Node("chatsk", r"$\hat{c}_i^{\rm s}$", 17.625, 2, observed=True))
			if fix_redshifts:
				pgm.add_node(daft.Node("zhatsk", r"$\hat{z}_i^{\rm s}$", 18.625, 2, observed=True, plot_params=p_color))
				pgm.add_node(daft.Node("q0hat", r"$\hat{q}_0$", 20.2, 2, observed=True))
			else:
				pgm.add_node(daft.Node("zhatsk", r"$\hat{z}_i^{\rm s}$", 18.625, 2, observed=True))
				pgm.add_node(daft.Node("q0hat", r"$\hat{q}_0$", 20.125, 2, observed=True))
		else:
			pgm.add_node(daft.Node("mhatsi", r"$\hat{m}_i^{\rm s}$", 12.25, 2, observed=True))
			pgm.add_node(daft.Node("mhatsk", r"$\hat{m}_i^{\rm s}$", 14.425, 2, observed=True))
			pgm.add_node(daft.Node("xhatsk", r"$\hat{x}_i^{\rm s}$", 15.425, 2, observed=True))
			pgm.add_node(daft.Node("chatsk", r"$\hat{c}_i^{\rm s}$", 16.425, 2, observed=True))
			if fix_redshifts:
				pgm.add_node(daft.Node("zhatsk", r"$\hat{z}_i^{\rm s}$", 17.425, 2, observed=True, plot_params=p_color))
				pgm.add_node(daft.Node("q0hat", r"$\hat{q}_0$", 19, 2, observed=True))
			else:
				pgm.add_node(daft.Node("zhatsk", r"$\hat{z}_i^{\rm s}$", 17.425, 2, observed=True))
				pgm.add_node(daft.Node("q0hat", r"$\hat{q}_0$", 19.925, 2, observed=True))

	# Edges.
	pgm.add_edge("pdhat0", "dhat0")
	pgm.add_edge("pmhat0", "mhat0")
	if pgm == pgm_full:
		pgm.add_edge("pphat0", "phat0")
		pgm.add_edge("pZhat0", "Zhat0")
	else:
		pgm.add_edge("phat0", "pm0j")
		pgm.add_edge("Zhat0", "Zhat0bend", directed=False)
	pgm.add_edge("pmhati", "mhati")
	if pgm == pgm_full:
		pgm.add_edge("pphati", "phati")
		pgm.add_edge("pZhati", "Zhati")
	else:
		pgm.add_edge("phati", "pmij")
		pgm.add_edge("Zhati", "Zhatibend", directed=False)
	if pgm == pgm_r16:
		pgm.add_edge("paxhat", "axhat")
		pgm.add_edge("pmhatsi", "mhatsi")
	else:
		if pgm == pgm_full:
			pgm.add_edge("pmxchatsi", "mhatsi")
			pgm.add_edge("pmxchatsi", "xhatsi")
			pgm.add_edge("pmxchatsi", "chatsi")
		else:
			pgm.add_edge("pmhatsi", "mhatsi")
		pgm.add_edge("pq0hat", "q0hat")
		pgm.add_edge("pmxchatsk", "mhatsk")
		pgm.add_edge("pmxchatsk", "xhatsk")
		pgm.add_edge("pmxchatsk", "chatsk")
		if fix_redshifts:
			pgm.add_edge("zhatskbend", "pmsk")
			pgm.add_edge("zhatsk", "zhatskbend", directed=False)
		else:
			pgm.add_edge("pzhatsk", "zhatsk")

	# Plates.
	pgm.add_plate(daft.Plate([2.0, 1.5, 4, 4], label=r"$0 \le j < n^c_i^{\rm s}$", \
		                     shift=-0.1, rect_params={"ec": "r"}))
	pgm.add_plate(daft.Plate([7.0, 1.5, 4, 4], label=r"$0 \le j < n^c_i^{\rm s}$", \
		                     shift=-0.1, rect_params={"ec": "r"}))
	pgm.add_plate(daft.Plate([0.325, 1.0, 5.875, 5.5], label=r"$0 \le i < n^{\rm ch,\, anc}$", \
		                     shift=-0.1, rect_params={"ec": "b"}))
	if pgm == pgm_full:
		pgm.add_plate(daft.Plate([6.75, 1.0, 7.625, 5.5], label=r"$n^{\rm ch,\, anc} \le i < n^{\rm ch,\, tot}$", \
			                     shift=-0.1, rect_params={"ec": "b"}))
		pgm.add_plate(daft.Plate([14.925, 1.0, 4.4, 5.5], label=r"$0 \le i < n^{\rm s,\,HF}$", \
			                     shift=-0.1, rect_params={"ec": "b"}))
	else:
		pgm.add_plate(daft.Plate([6.75, 1.0, 6.625, 5.5], label=r"$n^{\rm ch,\, anc} \le i < n^{\rm ch,\, tot}$", \
			                     shift=-0.1, rect_params={"ec": "b"}))
		if pgm != pgm_r16:
			pgm.add_plate(daft.Plate([13.725, 1.0, 4.4, 5.5], label=r"$0 \le i < n^{\rm s,\,HF}$", \
				                     shift=-0.1, rect_params={"ec": "b"}))

	# Render.
	pgm.render()

# Save.
if fix_redshifts:
	pgm_smf.figure.savefig("bhm_plot_fixed_z.pdf")
	pgm_full.figure.savefig("bhm_plot_full_fixed_z.pdf")
else:
	pgm_smf.figure.savefig("bhm_plot.pdf")
	pgm_full.figure.savefig("bhm_plot_full.pdf")
	pgm_r16.figure.savefig("bhm_plot_r16.pdf")
