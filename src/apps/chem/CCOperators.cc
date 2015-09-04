/*
 * CCOperators.cc
 *
 *  Created on: Jul 6, 2015
 *      Author: kottmanj
 */

#include "CCOperators.h"

namespace madness{

static double unitfunction(const coord_3d &r){
	return 1.0;
}

real_function_3d CC_Intermediates::make_density(const vecfuncT &bra,
		const vecfuncT &ket) const {
	if (bra.size() != ket.size())
		error(
				"error in make density: unequal sizes ("
				+ stringify(bra.size()) + " and "
				+ stringify(ket.size()) + ")");
	if (bra.empty())
		error("error in make_density: bra_element is empty");
	// make the density
	real_function_3d density = real_factory_3d(world);
	for (size_t i = 0; i < bra.size(); i++)
		density += bra[i] * ket[i];
	density.truncate();
	return density;
}

std::vector<vecfuncT> CC_Intermediates::make_exchange_intermediate(const vecfuncT &bra,
		const vecfuncT &ket) const {
	if (bra.size() != ket.size() or bra.empty())
		error(
				"in make_exchange_intermediate, bra and ket empty or unequal sizes:\n bra_size: "
				+ stringify(bra.size()) + ", ket_size: "
				+ stringify(ket.size()));
	std::vector<vecfuncT> EX;
	EX.resize(bra.size());
	for (size_t i = 0; i < bra.size(); i++) {
		EX[i].resize(ket.size());
		for (size_t j = 0; j < ket.size(); j++) {
			EX[i][j] = (*poisson)(bra[j] * ket[i]);
		}
		truncate(world, EX[i]);
	}
	return EX;
}

/// Calculates two electron integrals
/// <ij|g|kl>
Tensor<double> CC_Intermediates::make_two_electron_integrals_hf() const {
	Tensor<double> result(mo_bra_.size(), mo_bra_.size(), mo_ket_.size(),
			mo_ket_.size());
	for (size_t i = 0; i < mo_bra_.size(); i++) {
		for (size_t j = 0; j < mo_bra_.size(); j++) {
			for (size_t k = 0; k < mo_ket_.size(); k++) {
				for (size_t l = 0; l < mo_ket_.size(); l++) {
					result(i, j, k, l) = (mo_bra_[i] * mo_ket_[k]).inner(
							exchange_intermediate_[l][j]);
				}
			}
		}
	}
	return result;
}
/// <ij|g|k\tau_l>
Tensor<double> CC_Intermediates::make_two_electron_integrals_mixed_t1(
		const vecfuncT &tau) const {
	Tensor<double> result(mo_bra_.size(), mo_bra_.size(), mo_ket_.size(),
			tau.size());
	for (size_t i = 0; i < mo_bra_.size(); i++) {
		for (size_t j = 0; j < mo_bra_.size(); j++) {
			for (size_t k = 0; k < mo_ket_.size(); k++) {
				for (size_t l = 0; l < tau.size(); l++) {
					result(i, j, k, l) = (mo_bra_[i] * mo_ket_[k]).inner(
							perturbed_exchange_intermediate_[l][j]);
				}
			}
		}
	}
	return result;
}
// <ij|g|\tau_k \tau_l>
Tensor<double> CC_Intermediates::make_two_electron_integrals_t1(const vecfuncT &tau) const {
	Tensor<double> result(mo_bra_.size(), mo_bra_.size(), tau.size(),
			tau.size());
	for (size_t i = 0; i < mo_bra_.size(); i++) {
		for (size_t j = 0; j < mo_bra_.size(); j++) {
			for (size_t k = 0; k < tau.size(); k++) {
				for (size_t l = 0; l < tau.size(); l++) {
					result(i, j, k, l) = (mo_bra_[i] * tau[k]).inner(
							perturbed_exchange_intermediate_[l][j]);
				}
			}
		}
	}
	return result;
}

double CC_Operators::compute_mp2_pair_energy(CC_Pair &pair)const{

	// this will be the bra space
	real_function_6d eri = TwoElectronFactory(world).dcut(parameters.lo);
	real_function_6d ij_g =
			CompositeFactory<double, 6, 3>(world).particle1(
					copy(mo_bra_[pair.i])).particle2(
							copy(mo_bra_[pair.j])).g12(eri);
	real_function_6d ji_g =
			CompositeFactory<double, 6, 3>(world).particle1(
					copy(mo_bra_[pair.i])).particle2(
							copy(mo_bra_[pair.j])).g12(eri);

	// compute < ij | g12 | psi >
	const double ij_g_uij = inner(pair.function, ij_g);
	if (world.rank() == 0)
		printf("<ij | g12       | psi^1>  %12.8f\n", ij_g_uij);

	// compute < ji | g12 | psi > if (i/=j)
	const double ji_g_uij = (pair.i == pair.j) ? 0 : inner(pair.function, ji_g);
	if (world.rank() == 0)
		printf("<ji | g12       | psi^1>  %12.8f\n", ji_g_uij);

	// the singlet and triplet triplet pair energies
	if (pair.i == pair.j) {
		pair.e_singlet = ij_g_uij + pair.ij_gQf_ij;
		pair.e_triplet = 0.0;
	} else {
		pair.e_singlet = (ij_g_uij + pair.ij_gQf_ij)
								+ (ji_g_uij + pair.ji_gQf_ij);
		pair.e_triplet = 3.0
				* ((ij_g_uij - ji_g_uij) + (pair.ij_gQf_ij - pair.ji_gQf_ij));
	}

	// print the pair energies
	if (world.rank() == 0) {
		printf("current energy %2d %2d %12.8f %12.8f\n", pair.i, pair.j,
				pair.e_singlet, pair.e_triplet);
	}

	// return the total energy of this pair
	return pair.e_singlet + pair.e_triplet;
}

// The Fock operator is partitioned into F = T + Vn + R
// the fock residue R= 2J-K for closed shell is computed here
// J_i = \sum_k <k|r12|k> |tau_i>
// K_i = \sum_k <k|r12|tau_i> |k>
vecfuncT CC_Operators::fock_residue_closed_shell(const CC_Singles &singles) const {
	vecfuncT tau(singles.size());
	for(size_t x=0;x<tau.size();x++) tau[x]=singles[x].function();
	CC_Timer timer_J(world,"J");
	vecfuncT J = mul(world, intermediates_.get_hartree_potential(), tau);
	truncate(world, J);
	scale(world, J, 2.0);
	timer_J.info();
	CC_Timer timer_K(world,"K");
	vecfuncT K;
	for (size_t i = 0; i < tau.size(); i++) {
		real_function_3d tmp = real_factory_3d(world);
		vecfuncT vectmp = mul(world,
				intermediates_.get_perturbed_exchange_intermediate()[i],
				mo_ket_);
		for (size_t j = 0; j < tau.size(); j++)
			tmp += vectmp[j];
		tmp.truncate();
		K.push_back(tmp);
	}
	truncate(world, K);
	scale(world, K, -1);
	timer_K.info();
	return add(world, J, K);
}

// The coulomb Term of the S3C diagram: Positive sign
// \     /
//  \---/  = 2Q\sum_j(<j|g12|tau_j>)|i>
//  _\_/_
vecfuncT CC_Operators::S3c(const CC_Singles &singles) const {
	vecfuncT tau(singles.size());
	for(size_t x=0;x<tau.size();x++) tau[x]=singles[x].function();
	vecfuncT result = mul(world,
			intermediates_.get_perturbed_hartree_potential(), mo_ket_);
	Q(result);
	truncate(world, result);
	scale(world, result, 2.0);
	return result;
}

// The Exchange Term of the S3C diagram: Negative sign
// \  /
//  \/...   = -Q\sum_j(<j|g12|i>|tau_j>)
//     / \
//    _\_/_
vecfuncT CC_Operators::S3c_X(const CC_Singles &singles) const {
	vecfuncT result;
	vecfuncT tau(singles.size());
	for(size_t x=0;x<tau.size();x++) tau[x]=singles[x].function();
	for (size_t i = 0; i < tau.size(); i++) {
		real_function_3d tmp = real_factory_3d(world);
		vecfuncT vectmp = mul(world,
				intermediates_.get_exchange_intermediate()[i], tau);
		for (size_t j = 0; j < tau.size(); j++)
			tmp += vectmp[j];
		tmp.truncate();
		result.push_back(tmp);
	}
	Q(result);
	truncate(world, result);
	scale(world, result, -1.0);
	return result;
}

/// The S5b term
//[i]    [Q]
// \     /....
//  \   /   / \
//  _\_/_  _\_/_
// 2\sum_k <k|g|\tau_k> |\tau_i>
// No Q is applied yet !
vecfuncT CC_Operators::S5b(const CC_Singles &singles) const {
	vecfuncT tau(singles.size());
	for(size_t x=0;x<tau.size();x++) tau[x]=singles[x].function();
	vecfuncT result = mul(world,
			intermediates_.get_perturbed_hartree_potential(), mo_ket_);
	truncate(world, result);
	scale(world, result, 2.0);
	return result;
}

/// The S5b Exchange Term
//[i]         [Q]
// \     ...../
//  \   /\   /
//  _\_/  \_/_
// -\sum_k <k|g|\tau_i> |\tau_k>
// No Q is applied yet !
vecfuncT CC_Operators::S5b_X(const CC_Singles &singles) const {
	vecfuncT tmp;
	vecfuncT tau(singles.size());
	for(size_t x=0;x<tau.size();x++) tau[x]=singles[x].function();
	vecfuncT result = zero_functions_compressed<double, 3>(world,
			(tau.size()));
	for (size_t i = 0; i < tau.size(); i++) {
		tmp = mul(world,
				intermediates_.get_perturbed_exchange_intermediate()[i],
				tau);
		for (size_t k = 0; k < tau.size(); k++) {
			result[i] += tmp[k];
		}
	}
	truncate(world, result);
	scale(world, result, -1);
	return result;
}

/// The S5c term
//[Q]    [i]
// \     /....
//  \   /   / \
//  _\_/_  _\_/_
// -2\sum_kl <kl|g|i\tau_l> |\tau_k>
// No Q is applied yet !
// May use alteriative algorithm with perturbed density intermediate
vecfuncT CC_Operators::S5c(const CC_Singles&tau) const {
	vecfuncT result = zero_functions_compressed<double, 3>(world,
			tau.size());
	for (size_t i = 0; i < mo_bra_.size(); i++) {
		for (size_t k = 0; k < mo_bra_.size(); k++) {
			for (size_t l = 0; l < mo_bra_.size(); l++) {
				result[i] += intermediates_.get_integrals_mixed_t1()(k, l,
						i, l) * tau[k].function();
			}
		}
	}
	truncate(world, result);
	scale(world, result, -2.0);
	return result;
}

/// The S5c_X echange term
//[Q]         [i]
// \     ...../
//  \   /\   /
//  _\_/  \_/_
// -\sum_kl <lk|g|i\tau_l> |\tau_k>
// No Q is applied yet !
vecfuncT CC_Operators::S5c_X(const CC_Singles &tau) const {
	vecfuncT result = zero_functions_compressed<double, 3>(world,
			tau.size());
	for (size_t i = 0; i < mo_bra_.size(); i++) {
		for (size_t k = 0; k < mo_bra_.size(); k++) {
			for (size_t l = 0; l < mo_bra_.size(); l++) {
				result[i] += intermediates_.get_integrals_mixed_t1()(l, k,
						i, l) * tau[k].function();
			}
		}
	}
	truncate(world, result);
	scale(world, result, -1.0);
	return result;
}

/// The S6+X Term
// \    /\    /...
//  \  /  \  /   /\
//  _\/_  _\/_  _\/_
// -Q \sum_kl 2<kl|g|\tau_k\tau_i> |\tau_l> - \sum_kl <kl|g|\taui\tau_k> |\tau_l>
// Q is not applied yet!
vecfuncT CC_Operators::S6(const CC_Singles &tau) const {
	vecfuncT result = zero_functions_compressed<double, 3>(world,
			tau.size());
	for (size_t i = 0; i < tau.size(); i++) {
		for (size_t k = 0; k < mo_bra_.size(); k++) {
			for (size_t l = 0; l < mo_bra_.size(); l++) {
				result[i] += (-2
						* intermediates_.get_integrals_t1()(k, l, k, i)
						- intermediates_.get_integrals_t1()(k, l, i, k))
						* tau[l].function();
			}
		}
	}
	truncate(world, result);
	return result;
}

/// CC2 singles diagrams with 6d functions as input
/// Use GFInterface in function_interface.h as kernel (f*g) and do not reconstruct \tau = f12u(1,2) if possible
/// Since the correlation factor of CC2 has Slater form like in MP2: g12f12 = g12(1-exp(-mu*r12)/r12) = g12 - exp(-mu*r12)/r12 = Coulomb_Operator - BSH_Operator(mu)

/// S2b + X Term
// [i]   [Q]
//  \    /....
//   \  /    /\
//  __\/_____\/__
// -\sum_k(2.0 <k(2)|gf|u_ki(2,1)>_2 - <k(2)|gf|u_ki(1,2)>_2)
// =-\sum_k(2.0 <k(2)|gf|u_ik(1,2)>_2 - <k(2)|gf|u_ki(1,2)>_2)

/// Current procedure:
/// use g12 = \int \delta(1-3) g32 d3
/// <k(2)|g12|u(1,2)> = \int d2[ g12x(1,2) ] with x(1,2) = k(2)u(1,2)
/// = int d2 [ int d3[ \delta(1-3) g32 ] x(1,2) ]
/// = \int d3[\delta(1-3) \int d2 [ g32 x(1,2) ] ]
/// = \int d3[\delta(1-3) h(1,3)] with h(1,3) = \int d2 g23 x(1,2)
vecfuncT CC_Operators::S2b(const Pairs<CC_Pair> u) const {
	// debug
	u(0,0).function.print_size("\t Doubles which enter S2b Function");

			real_function_3d unity = real_factory_3d(world).f(unitfunction);
	vecfuncT result(mo_ket_.size());
	for (size_t i = 0; i < mo_ket_.size(); i++) {
		real_function_3d resulti = real_factory_3d(world);
		for (size_t k = 0; k < mo_ket_.size(); k++) {
			// calculate x(1,2) from u(1,2) and k(2), --> F.A.B uses multiply(copy(f),copy(bra) ...) deep copy of functions (dont know why) ... check if this is necessary
			real_function_6d xik = (multiply(copy(u(i, k).function), mo_bra_[k], 2)).truncate();
			real_function_6d xki = (multiply(copy(u(k, i).function), mo_bra_[k], 2)).truncate();
			// calculate the convolution with fg = 1/(2gamma)*(Coulomb - 4pi*BSH(gamma))
			real_function_6d hik= apply_gf(xik,2);;
			real_function_6d hki= apply_gf(xki,2);
			// Make the projection to 3D with the unit function
			real_function_3d resultik = hik.project_out(unity, 1); // the 1 means that the second particle is integrated out (?) , can be only 1 or 0
			real_function_3d resultki = hki.project_out(unity, 1);
			resultik.truncate();
			resultki.truncate();
			if(parameters.debug and i==k){
				real_function_3d diff = (resultik - resultki);
				if(fabs(diff.norm2())>FunctionDefaults<6>::get_thresh()){
					if(world.rank()==0) std::cout << "\n\n!!!!Error in S2b for equal indices -> Exchange part is not the same!!!!\n\n" << std::endl;
				}
			}
			resulti += (2.0 * resultik - resultki);
		}
		result[i] = resulti;
	}

	Q(result);
	truncate(world, result);
	if(parameters.debug and world.rank()==0){
		std::cout << "S2b potential norms: ";
		for(auto x:result) std::cout << x.norm2() << " " << std::endl;
	}
	u(0,0).function.print_size("\t Doubles which leave S2b Function");
	return result;


}

/// S2c + X Term
// [Q]   [i]
//  \    /....
//   \  /    /\
//  __\/_____\/__
/// = \sum_kl 2<k(3)l(4)|g34|t_kl(3,1) i(4)> - <k(3)l(4)|g34|t_kl(4,1) i(3)>
// current procedure:
// make t_kl = f12*u(12) with the composite factory
// make intermediate: lgi(3) = <l|g|i> and kgi(4) = <k|g|i>
// process intermediate: klgi = k*lgi and lkgi = l*kgi
// Project out the intermediate from the t_kl function  <klgi|t_kl>_1 and <lkgi|t_kl>_1
// return 2.0*<klgi|t_kl>_1 - <lkgi|t_kl>_1

vecfuncT CC_Operators::S2c(const Pairs<CC_Pair> u) const {
	u(0,0).function.print_size("\t Doubles which enter S2c Function");
	vecfuncT result;
	for(size_t i=0;i<mo_ket_.size();i++){
		real_function_3d resulti = real_factory_3d(world);
		for(size_t k=0;k<mo_ket_.size();k++){
			for(size_t l=0;l<mo_ket_.size();l++){
				// make pair function t_kl
				CC_Timer make_tau(world,"Creating cuspy pair: f12*u(1,2)");
				real_function_6d tkl =  CompositeFactory<double, 6, 3>(world).g12(corrfac.f()).ket(copy(u(k,l).function));
				tkl.fill_tree().truncate().reduce_rank();
				make_tau.info();
				tkl.print_size("f12*u(12)");
				// make the exchange intermediates
				real_function_3d klgi = (mo_bra_[k]*intermediates_.get_exchange_intermediate()[i][l]).truncate();
				real_function_3d lkgi = (mo_bra_[l]*intermediates_.get_exchange_intermediate()[i][k]).truncate();
				// Project out

				// test for debug
				real_function_6d test = copy(tkl);
				real_function_3d klgitkl = tkl.project_out(klgi,0);
				// better check if t_kl changes during the procedure
				{
					if(world.rank()==0) std::cout << "S2c potential debug output:\n";
					tkl.print_size("pair function after");
					test.print_size("pair function before (should be the same");
				}
				real_function_3d lkgitkl = tkl.project_out(lkgi,0); // 0 means first particle coordinates are intergrated
				{
					if(world.rank()==0) std::cout << "S2c potential debug output:\n";
					tkl.print_size("pair function after");
					test.print_size("pair function before (should be the same");
				}
				resulti += (2.0*klgitkl - lkgitkl).truncate();
				Q(resulti);
			}
		}
		resulti.print_size("S2c_"+stringify(i)+" potential");
		result.push_back(resulti);
	}
	u(0,0).function.print_size("\t Doubles which leave S2c Function");
	return result;
}
/// The S4a + X diagram
//[Q]       [i]
// \    ..../.....
//  \  /\  /     /\
//  _\/_ \/______\/_
/// -Q\sum (2<kl|g|\tau_il>|\tau_k> - <kl|g|\tau_ik>|\tau_l>)  : <kl|g|\tau_il>|\tau_k> = <k>
vecfuncT CC_Operators::S4a(const Pairs<CC_Pair> u, const CC_Singles & tau) const {
	vecfuncT result(mo_ket_.size());
	for (size_t i = 0; i < mo_ket_.size(); i++) {
		real_function_3d resulti = real_factory_3d(world);
		for (size_t k = 0; k < mo_ket_.size(); k++) {
			for (size_t l = 0; l < mo_ket_.size(); l++) {
				// Coulomb Part of f12g12 = g12 - BSH
				{
					real_function_6d eri = TwoElectronFactory(world).dcut(
							FunctionDefaults<3>::get_thresh());
					real_function_6d kl_g = CompositeFactory<double, 6, 3>(
							world).particle1(copy(mo_bra_[k])).particle2(
									copy(mo_bra_[l])).g12(eri);
					resulti -= (2.0 * inner(u(i, l).function, kl_g) * tau[k].function()
					                                                      - inner(u(i, k).function, kl_g) * tau[l].function());
					resulti.truncate();
				}
				// BSH part of f12g12
				{
					real_function_6d bsh_kernel =
							TwoElectronFactory(world).BSH().dcut(
									FunctionDefaults<3>::get_thresh());
					real_function_6d kl_bsh =
							CompositeFactory<double, 6, 3>(world).particle1(
									copy(mo_bra_[k])).particle2(
											copy(mo_bra_[l])).g12(bsh_kernel);
					resulti -= (2.0 * inner(u(i, l).function, kl_bsh) * tau[k].function()
					                                                        - inner(u(i, k).function, kl_bsh) * tau[l].function());
					resulti.truncate();
				}
			}
		}
		resulti.print_size("S4a_"+stringify(i)+" potential");
		result[i] = resulti;
	}
	Q(result);
	return result;
}

/// The S4b -> Merge this with S2c later to save time
//[i]       [Q]
// \    ..../.....
//  \  /\  /     /\
//  _\/_ \/______\/_
/// -Q\sum_{kl} (2<k(3)l(4)|g34f14|\tau_{i}(3)u_{kl}(1,4)>  // exchange part - <k(4)l(3)|g34f14|\tau_i(3)u_{lk}(1,4)>)
/// =-Q\sum_{kl} (2<k(3)l(4)|g34f14|\tau_{i}(3)u_{kl}(1,4)>  // exchange part - <k(4)l(3)|g34f14|\tau_i(3)u_{kl}(4,1)>)
// current procedure
// make intermediates: <k|\tau_i> and <l|g|\tau_i> -> get the perturbed_exchange_intermediate
// process intermedaites: lkgi = l*<k|g|tau_i> amd klgi = k*<l|g|\tau_i> (k and l are from the bra space)
// Project out:  <lkgi|tkl>_2 and <klgi|tkl>_1
// return -2.0*Q(<lkgi|tkl>_2) + Q(<klgi|tkl>_1)
vecfuncT CC_Operators::S4b(const Pairs<CC_Pair> u, const CC_Singles & tau) const {
	vecfuncT result;
	for(size_t i=0;i<mo_ket_.size();i++){
		real_function_3d resulti = real_factory_3d(world);
		for(size_t k=0;k<mo_ket_.size();k++){
			for(size_t l=0;l<mo_ket_.size();l++){
				// make the cuspy pair
				CC_Timer make_tau(world,"Creating cuspy pair: f12*u(1,2)");
				real_function_6d tkl =  CompositeFactory<double, 6, 3>(world).g12(corrfac.f()).ket(u(k,l).function);
				tkl.fill_tree().truncate().reduce_rank();
				make_tau.info();
				tkl.print_size("f12*u(12)");

				// make intermediates
				real_function_3d lkgi = (mo_bra_[l]*intermediates_.get_perturbed_exchange_intermediate()[i][k]).truncate();
				real_function_3d klgi = (mo_bra_[k]*intermediates_.get_perturbed_exchange_intermediate()[i][l]).truncate();

				// project out (1 is particle 2 and 0 is particle 1)
				real_function_3d lkgitkl = tkl.project_out(lkgi,1);
				real_function_3d klgitkl = tkl.project_out(klgi,0);

				resulti += (-2.0*lkgitkl+klgitkl).truncate();
				Q(resulti);
			}
		}
		resulti.print_size("S4b_"+stringify(i)+" potential");
		result.push_back(resulti);
	}
	Q(result);
	return result;
}

/// The S4c + X + X + X + X Diagrams -> merge this with S2c later to save time
//            [i]   [Q]
//   .......   \    /
//  /\     /\   \  /
// _\/_   _\/____\/_
/// Q\sum_{kl}[ 4*<k(3)l(4)|g34 f14| \tau_k(3) u_{il}(1,4)> - 2* <k(3)l(4)|g34 f14|\tau_k(4) u_{il}(1,3)>
/// -           2*<k(3)l(4)|g34 f14| \tau_k(3) u_{il}(4,1)>	    +<k(3)l(4)|g34 f14|\tau_k(4) u_{il}(3,1)>  ]

// current procedure:
// make cuspy function: til = uil*f12
// make intermediates: lkgk= l*<k|g|\tau_k> and klgk= k*<l|g|\tau_k>
// Project out:
// 1st term: <lkgk|til>_2
// 2nd term: <klgk|til>_2
// 3rd term: <lkgk|til>_1
// 4th term: <klgk|til>_1

vecfuncT CC_Operators::S4c(const Pairs<CC_Pair> u, const CC_Singles & tau) const {
	vecfuncT result;
	for(size_t i=0;i<mo_ket_.size();i++){
		real_function_3d resulti = real_factory_3d(world);
		for(size_t k=0;k<mo_ket_.size();k++){
			for(size_t l=0;l<mo_ket_.size();l++){
				// make the cuspy pair
				CC_Timer make_tau(world,"Creating cuspy pair: f12*u(1,2)");
				real_function_6d til =  CompositeFactory<double, 6, 3>(world).g12(corrfac.f()).ket(u(i,l).function);
				til.fill_tree().truncate().reduce_rank();
				make_tau.info();
				til.print_size("f12*u(12)");

				// get intermediates
				real_function_3d lkgk = (mo_bra_[l]*intermediates_.get_perturbed_exchange_intermediate()[k][k]).truncate();
				real_function_3d klgk = (mo_bra_[l]*intermediates_.get_perturbed_exchange_intermediate()[k][l]).truncate();

				// project out
				real_function_3d term1 = til.project_out(lkgk,1);
				real_function_3d term2 = til.project_out(klgk,1);
				real_function_3d term3 = til.project_out(lkgk,0);
				real_function_3d term4 = til.project_out(klgk,0);

				resulti += (4.0*term1 -2.0*term2 - 2.0*term3 + term4).truncate();
				Q(resulti);
			}
		}
		resulti.print_size("S4c_"+stringify(i)+" potential");
		result.push_back(resulti);
	}
	Q(result);
	return result;
}

/// The 6D Fock residue on the cusp free pair function u_{ij}(1,2) is: (2J - Kn - Un)|u_{ij}>
real_function_6d CC_Operators::fock_residue_6d(const CC_Pair &u) const {
	const double eps = get_epsilon(u.i, u.j);
	// make the coulomb and local Un part with the composite factory
	real_function_3d local_part = (2.0
			* intermediates_.get_hartree_potential()
			+ nemo.nuclear_correlation->U2());
	local_part.print_size("vlocal");
	u.function.print_size("u");

	// Contruct the BSH operator in order to screen

	real_convolution_6d op_mod = BSHOperator<6>(world, sqrt(-2 * eps),
			parameters.lo, parameters.thresh_bsh_6D);
	// apparently the modified_NS form is necessary for the screening procedure
	op_mod.modified() = true;
	// Make the CompositeFactory
	real_function_6d vphi =
			CompositeFactory<double, 6, 3>(world).ket(copy(u.function)).V_for_particle1(
					copy(local_part)).V_for_particle2(copy(local_part));
	// Screening procedure
	vphi.fill_tree(op_mod);

	vphi.print_size("vlocal|u>");

	// the part with the derivative operators: U1
	for (int axis = 0; axis < 6; ++axis) {
		real_derivative_6d D = free_space_derivative<double, 6>(world,
				axis);
		// Partial derivative of the pari function
		const real_function_6d Du = D(u.function).truncate();

		// % operator gives division rest (modulo operator)
		if (world.rank() == 0)
			print("axis, axis^%3, axis/3+1", axis, axis % 3, axis / 3 + 1);
		const real_function_3d U1_axis = nemo.nuclear_correlation->U1(
				axis % 3);

		double tight_thresh = parameters.thresh_Ue;
		real_function_6d x;
		if (axis / 3 + 1 == 1) {
			x =
					CompositeFactory<double, 6, 3>(world).ket(Du).V_for_particle1(
							copy(U1_axis)).thresh(tight_thresh);

		} else if (axis / 3 + 1 == 2) {
			x =
					CompositeFactory<double, 6, 3>(world).ket(Du).V_for_particle2(
							copy(U1_axis)).thresh(tight_thresh);
		}
		x.fill_tree(op_mod);
		x.set_thresh(FunctionDefaults<6>::get_thresh());
		vphi += x;
		vphi.truncate().reduce_rank();
	}

	vphi.print_size("(Un + J1 + J2)|u>");

	// Exchange Part
	vphi = (vphi - K(u.function, u.i == u.j)).truncate().reduce_rank();
	vphi.print_size("(Un + J1 + J2 - K1 - K2)|U>");
	vphi.truncate();
	vphi.print_size("truncated: (Un + J1 + J2 - K1 - K2)|U>");
	return vphi;

}

/// Echange Operator on 3D function
/// !!!!Prefactor (-1) is not included
real_function_3d CC_Operators::K(const real_function_3d &f,const size_t &i, const bool hc)const{
	real_function_3d result = real_factory_3d(world);
	if(hc==true) MADNESS_EXCEPTION("ERROR in K, hc=true not implemented",1);

	for (std::size_t k = 0; k < mo_ket_.size(); ++k) {
		result += mo_ket_[k]*intermediates_.get_exchange_intermediate()[i][k];
	}

	// Sanity Check (expensive when not helium)
	if(mo_ket_.size()<3){
		real_function_3d result2 = real_factory_3d(world);
		// multiply rhs with R2orbitals (the bra space)
		vecfuncT R2rhs = mul(world, f, mo_bra_);
		for (std::size_t k = 0; k < mo_ket_.size(); ++k) {
			result2 += mo_ket_[k] * (*poisson)(R2rhs[k]);
		}
		double sanity = (result-result2).norm2();
		if(sanity < FunctionDefaults<3>::get_thresh()){
			if(world.rank()==0) std::cout << "Sanity Check of K passed\n";
		}else{
			if(world.rank()==0) std::cout << "Sanity Check of K NOT passed\n";
		}
	}

	return result;
}

/// Exchange Operator on Pair function: -(K(1)+K(2))u(1,2)
/// if i==j in uij then the symmetry will be exploited
/// !!!!Prefactor (-1) is not included here!!!!
real_function_6d CC_Operators::K(const real_function_6d &u,
		const bool symmetric) const {
	/// DEBUG
	if(world.rank()==0)std::cout << "Entering K" << std::endl;
	/// DEBUG END

	/// TEST IF THIS WILL WORK FOR THE += Operator
	real_function_6d result = real_factory_6d(world).compressed();
	// K(1) Part
	result += apply_K(u, 1);
	// K(2) Part
	if (symmetric)
		result += swap_particles(result);
	else
		result += apply_K(u, 2);

	return (result.truncate());
}

/// Exchange Operator on Pair function: -(K(1)+K(2))u(1,2)
/// K(1)u(1,2) = \sum_k <k(3)|g13|u(3,2)> |k(1)>
/// 1. X(3,2) = bra_k(3)*u(3,2)
/// 2. Y(1,2) = \int X(3,2) g13 d3
/// 3. result = Y(1,2)*ket_k(1)
/// !!!!Prefactor (-1) is not included here!!!!
real_function_6d CC_Operators::apply_K(const real_function_6d &u,
		const size_t &particle) const {
	/// DEBUG
	if(world.rank()==0)std::cout << "Entering apply_K" << std::endl;
	/// DEBUG END
	MADNESS_ASSERT(particle == 1 or particle == 2);
	poisson->particle() = particle;
	/// WARNING: CHECK IF THIS WORKS -> bc of the += operator later
	real_function_6d result = real_factory_6d(world).compressed();
	for (size_t k = 0; k < mo_ket_.size(); k++) {
		real_function_6d X = (multiply(copy(u), copy(mo_bra_[k]), particle)).truncate();
		real_function_6d Y = (*poisson)(X);
		result += multiply(copy(Y), copy(mo_ket_[k]), particle).truncate();
	}
	return result;
}

/// Apply Ue on a tensor product of two 3d functions: Ue(1,2) |x(1)y(2)> (will be either |ij> or |\tau_i\tau_j> or mixed forms)
/// The Transformed electronic regularization potential (Kutzelnigg) is R_{12}^{-1} U_e R_{12} with R_{12} = R_1*R_2
/// It is represented as: R_{12}^{-1} U_e R_{12} = U_e + R^-1[Ue,R]
/// where R^-1[Ue,R] = R^-1 [[T,f],R] (see: Regularizing the molecular potential in electronic structure calculations. II. Many-body
/// methods, F.A.Bischoff)
/// The double commutator can be evaluated as follows:  R^-1[[T,f],R] = -Ue_{local}(1,2)*(Un_{local}(1) - Un_{local}(2))
/// @param[in] x the 3D function for particle 1
/// @param[in] y the 3D function for particle 2
/// @param[in] i the first index of the current pair function (needed to construct the BSH operator for screening)
/// @param[in] j the second index of the current pair function
/// @param[out]  R^-1U_eR|x,y> the transformed electronic smoothing potential applied on |x,y> :
real_function_6d CC_Operators::apply_transformed_Ue(const real_function_3d x,
		const real_function_3d y, const size_t &i, const size_t &j, CC_Pair &u) const {
	real_function_6d Uxy = real_factory_6d(world);
	// Apply the untransformed U Potential
	const double eps = get_epsilon(i, j);
	Uxy = corrfac.apply_U(x, y, eps);

	// Get the 6D BSH operator in modified-NS form for screening
	real_convolution_6d op_mod = BSHOperator<6>(world, sqrt(-2 * eps),
			parameters.lo,
			parameters.thresh_Ue);
	op_mod.modified() = true;

	// make shure the thresh is high enough
	double tight_thresh = parameters.thresh_6D_tight;

	// Apply the double commutator R^{-1}[[T,f,R]
	for (size_t axis = 0; axis < 3; axis++) {
		// Make the local parts of the Nuclear and electronic U potentials
		const real_function_3d Un_local = nemo.nuclear_correlation->U1(
				axis);
		const real_function_3d Un_local_x = (Un_local * x).truncate();
		const real_function_3d Un_local_y = (Un_local * y).truncate();
		const real_function_6d Ue_local = corrfac.U1(axis);
		// Now add the Un_local_x part to the first particle of the Ue_local potential
		real_function_6d UeUnx = CompositeFactory<double, 6, 3>(world).g12(
				Ue_local).particle1(Un_local_x).particle2(copy(y)).thresh(
						tight_thresh);
		// Fill the Tree were it will be necessary
		UeUnx.fill_tree(op_mod);
		// Set back the thresh
		UeUnx.set_thresh(FunctionDefaults<6>::get_thresh());

		UeUnx.print_size("UeUnx");

		// Now add the Un_local_y part to the second particle of the Ue_local potential
		real_function_6d UeUny = CompositeFactory<double, 6, 3>(world).g12(
				Ue_local).particle1(copy(x)).particle2(Un_local_y).thresh(
						tight_thresh);
		// Fill the Tree were it will be necessary
		UeUny.fill_tree(op_mod);
		// Set back the thresh
		UeUny.set_thresh(FunctionDefaults<6>::get_thresh());

		UeUny.print_size("UeUny");

		// Construct the double commutator part and add it to the Ue part
		real_function_6d diff = (UeUnx - UeUny).scale(-1.0);
		diff.truncate();
		Uxy = (Uxy+diff).truncate();
	}

	// sanity check: <xy|R2 [T,g12] |xy> = <xy |R2 U |xy> - <xy|R2 g12 | xy> = 0
	real_function_6d tmp = CompositeFactory<double, 6, 3>(world).particle1(
			copy(mo_bra_[i])).particle2(copy(mo_bra_[j]));

	const double a = inner(Uxy, tmp);
	const real_function_3d xx = (x * x*nemo.nuclear_correlation -> square());
	const real_function_3d yy = (y * y*nemo.nuclear_correlation -> square());
	const real_function_3d gxx = (*poisson)(xx);
	const double aa = inner(yy, gxx);
	const double error = std::fabs(a - aa);
	if (world.rank() == 0) {
		printf("< phi0 | U_R   | phi0 >  %12.8f\n", a);
		printf("< phi0 | 1/r12 | phi0 >  %12.8f\n", aa);
		if (error > FunctionDefaults<6>::get_thresh())
			print("WARNING : Kutzelnigg's potential inaccurate (box size, thresh ?)");
		//if (error > FunctionDefaults<6>::get_thresh() * 10.0)
		//	MADNESS_EXCEPTION("Kutzelnigg's potential plain wrong (box size, thresh ?)", 1);
	}
	Uxy.print_size("Uphi0");

	return Uxy;
}

/// Apply the Exchange Commutator [K,f]|xy>
real_function_6d CC_Operators::apply_exchange_commutator(const real_function_3d &x, const real_function_3d &y,const std::string &type, const size_t &i, const size_t &j)const{
	MADNESS_ASSERT(
			type == "occupied" or type == "mixed" or type == "virtual");


	// make first part of commutator
	real_function_6d Kfxy = apply_Kf(x,y,type,i,j).truncate();

	// for sanity check:
	double expv_first_part = 0.0;
	double expv_second_part = 0.0;
	if(type=="occupied"){
		real_function_6d tmp = CompositeFactory<double, 6, 3>(world).particle1(
				copy(mo_bra_[i])).particle2(copy(mo_bra_[j]));
		expv_first_part = inner(Kfxy,tmp);
	}


	// make the second part of the commutator
	real_function_6d fKxy = apply_fK(x,y,type,i,j).truncate();

	// fot the sanity check
	if(type=="occupied"){
		real_function_6d tmp = CompositeFactory<double, 6, 3>(world).particle1(
				copy(mo_bra_[i])).particle2(copy(mo_bra_[j]));
		expv_second_part = inner(fKxy,tmp);
	}

	if(type=="occupied"){
		if(world.rank()==0){
			std::cout << "Apply [K,f]|x,y> sanity check:";
			std::cout <<  "\n<ij|Kf|ij> =" << expv_first_part;
			std::cout <<  "\n<ij|fK|ij> =" << expv_second_part;
		}
	}

	real_function_6d result = (Kfxy - fKxy);

	if(type=="occupied"){
		// sanity check: The Expectation value of the Kommutator must vanish (symmetry)
		// <0|[A,B]|0> = <0|AB|0> - <0|BA|0> since A=f and B=K -> both are hermitian
		real_function_6d tmp = CompositeFactory<double, 6, 3>(world).particle1(
				copy(mo_bra_[i])).particle2(copy(mo_bra_[j]));
		const double a = inner(result, tmp);
		if (world.rank() == 0) {
			printf("< nemo0 | R^2 R-1 [K,f] R | nemo0 >  %12.8f\n", a);
			if (std::fabs(a) > FunctionDefaults<6>::get_thresh())
				print("WARNING : exchange commutator inaccurate");
			if (std::fabs(a) > FunctionDefaults<6>::get_thresh() * 10.0)
				MADNESS_EXCEPTION("exchange commutator plain wrong", 1);
		}
	}

	return result;
}

/// Apply the Exchange operator on a tensor product multiplied with f12
/// !!! Prefactor of (-1) is not inclued in K here !!!!
real_function_6d CC_Operators::apply_Kf(const real_function_3d &x,
		const real_function_3d &y, const std::string &type, const size_t &i, const size_t &j) const {
	MADNESS_ASSERT(
			type == "occupied" or type == "mixed" or type == "virtual");

	bool symmetric = false;
	if(type == "occupied" and i==j) symmetric = true;
	if(type == "virtual" and i==j) symmetric = true;
	if(type == "mixed") symmetric = false;

	CC_Timer timer_f12xy(world,"Constructed f12|xy>");
	// First make the 6D function f12|x,y>
	real_function_6d f12xy = CompositeFactory<double, 6, 3>(world).g12(
			corrfac.f()).particle1(copy(x)).particle2(copy(y));
	f12xy.fill_tree().truncate().reduce_rank();
	timer_f12xy.info();
	// Apply the Exchange Operator
	real_function_6d result = K(f12xy, symmetric);
	return result.truncate();
}

/// Apply fK on a tensor product of two 3D functions
/// fK|xy> = fK_1|xy> + fK_2|xy>
/// @param[in] x, the first 3D function in |xy>
/// @param[in] y, the second 3D function in |xy>
/// @param[in] type, specifies if |xy> = |ij> (occupied), |xy> = |\tau_i,j> (mixed) or |xy> = |\tau_i\tau_j> (virtual)
/// @param[in] i, the number of the function: bsp if occupied then x_i = |i>, if virtual then x_i = \tau_i etc
/// @param[in] j , index of the second function
real_function_6d CC_Operators::apply_fK(const real_function_3d &x,
		const real_function_3d &y, const std::string &type, const size_t &i,
		const size_t &j) const {
	MADNESS_ASSERT(type == "occupied" or type == "mixed" or type == "virtual");

	const real_function_3d& phi_i = x;
	const real_function_3d& phi_j = y;

	const real_function_3d Kphi_i = K(phi_i,i,false);
	const real_function_3d Kphi_j = K(phi_j,j,false);

	real_function_6d fKphi0a = CompositeFactory<double, 6, 3>(world).g12(
			corrfac.f()).particle1(copy(phi_i)).particle2(copy(Kphi_j));
	fKphi0a.fill_tree().truncate();
	real_function_6d fKphi0b = CompositeFactory<double, 6, 3>(world).g12(
			corrfac.f()).particle1(copy(Kphi_i)).particle2(copy(phi_j));
	fKphi0b.fill_tree().truncate();

	real_function_6d fKphi0 = (fKphi0a + fKphi0b).truncate();
	return fKphi0;

}

/// swap particles 1 and 2

/// param[in]	f	a function of 2 particles f(1,2)
/// return	the input function with particles swapped g(1,2) = f(2,1)
real_function_6d CC_Operators::swap_particles(const real_function_6d& f) const {
	CC_Timer timer_swap(world,"swap particles");
	// this could be done more efficiently for SVD, but it works decently
	std::vector<long> map(6);
	map[0] = 3;
	map[1] = 4;
	map[2] = 5;	// 2 -> 1
	map[3] = 0;
	map[4] = 1;
	map[5] = 2;	// 1 -> 2
	timer_swap.info();
	return mapdim(f, map);
}

// Calculate the CC2 energy equation which is
// \omega = \sum_{ij} 2<ij|g|\tau_{ij}> - <ij|g|\tau_{ji}> + 2 <ij|g|\tau_i\tau_j> - <ij|g|\tau_j\tau_i>
// with \tau_{ij} = u_{ij} + Q12f12|ij> + Q12f12|\tau_i,j> + Q12f12|i,\tau_j> + Q12f12|\tau_i\tau_j>
double CC_Operators::get_CC2_correlation_energy() const {
	MADNESS_EXCEPTION("get_cc2_correlation_energy not implemented yet",1);
	return 0.0;
}
double CC_Operators::compute_cc2_pair_energy(const CC_Pair &u,
		const real_function_3d &taui, const real_function_3d &tauj) const {
	double omega = 0.0;
	const size_t i = u.i;
	const size_t j = u.j;
	// Contribution from u itself, we will calculate <uij|g|ij> instead of <ij|g|uij> and then just make the inner product (see also mp2.cc)
	{
		real_function_6d coulomb = TwoElectronFactory(world).dcut(
				FunctionDefaults<6>::get_thresh());
		real_function_6d g_ij =
				CompositeFactory<double, 6, 3>(world).particle1(
						copy(mo_bra_[i])).particle2(copy(mo_bra_[j])).g12(
								coulomb);
		real_function_6d g_ji =
				CompositeFactory<double, 6, 3>(world).particle1(
						copy(mo_bra_[j])).particle2(copy(mo_bra_[i])).g12(
								coulomb);
		const double uij_g_ij = inner(u.function, g_ij);
		const double uij_g_ji = inner(u.function, g_ji); // =uji_g_ij
		omega += 2.0 * uij_g_ij - uij_g_ji;
	}
	// Contribution from the mixed f12(|\tau_i,j>+|i,\tau_j>) part
	{
		omega += 2.0*make_ijgQfxy(u.i,u.j,mo_ket_[i],tauj);
		omega += 2.0*make_ijgQfxy(u.i,u.j,taui,mo_ket_[j]);
		omega -= make_ijgQfxy(u.j,u.i,mo_ket_[i],tauj);
		omega -= make_ijgQfxy(u.j,u.i,taui,mo_ket_[j]);
	}
	// Contribution from the f12|ij> part, this should be calculated in the beginning
	{
		omega += (2.0*u.ij_gQf_ij - u.ji_gQf_ij );
	}
	// Contribution from the f12|\tau_i\tau_j> part
	{
		omega += 2.0*make_ijgQfxy(u.i,u.j,taui,tauj);
		omega -= make_ijgQfxy(u.i,u.j,tauj,taui);
	}
	// Singles Contribution
	{
		// I should use intermediates later because the t1 integrals are also needed for the CC2 potential
		//omega += 2.0*intermediates_.get_integrals_t1()(u.i,u.j,u.i,u.j); //<ij|g|\taui\tauj>
		omega += 2.0*make_ijgxy(u.i,u.j,taui,tauj);
		//omega -= intermediates_.get_integrals_t1()(u.i,u.j,u.j,u.i);     //<ij|g|\tauj\taui>
		omega -= make_ijgxy(u.i,u.j,tauj,taui);
	}
	return omega;
}

/// General Function to make the intergral <ij|gQf|xy>
double CC_Operators::make_ijgQfxy(const size_t &i, const size_t &j, const real_function_3d &x, const real_function_3d &y)const{
	// Q12 = I12 - O1 - O2 + O12
	real_function_3d jy = mo_bra_[j]*y;
	real_function_3d ix = mo_bra_[i]*x;
	// I12 Part:
	double ijgfxy = (ix).inner(apply_gf(jy));
	// O1 Part
	double ijgO1fxy =0.0;
	for(size_t k=0;k<mo_ket_.size();k++){
		real_function_3d igk = intermediates_.get_exchange_intermediate()[k][i];
		real_function_3d kfx = (*f12op)(mo_bra_[k]*x);
		real_function_3d igkkfx = (igk*kfx).truncate();
		ijgO1fxy += jy.inner(igkkfx);
	}
	// O2 Part
	double ijgO2fxy =0.0;
	for(size_t k=0;k<mo_ket_.size();k++){
		real_function_3d jgk = intermediates_.get_exchange_intermediate()[k][j];
		real_function_3d kfy = (*f12op)(mo_bra_[k]*y);
		real_function_3d jgkkfy = (jgk*kfy).truncate();
		ijgO2fxy += ix.inner(jgkkfy);
	}
	// O12 Part
	double ijgO12fxy = 0.0;
	for(size_t k=0;k<mo_ket_.size();k++){
		real_function_3d igk = intermediates_.get_exchange_intermediate()[k][i];
		real_function_3d kfx = (*f12op)(mo_bra_[k]*x);
		for(size_t l=0;l<mo_ket_.size();l++){
			double ijgkl = igk.inner(mo_bra_[j]*mo_ket_[l]);
			double klfxy = kfx.inner(mo_bra_[l]*y);
			ijgO12fxy += ijgkl*klfxy;
		}
	}

	return (ijgfxy - ijgO1fxy - ijgO2fxy + ijgO12fxy);
}

double CC_Operators::make_ijgfxy(const size_t &i, const size_t &j, const real_function_3d &x, const real_function_3d &y)const{
	real_function_3d jy = mo_bra_[j]*y;
	real_function_3d ix = mo_bra_[i]*x;
	// I12 Part:
	double ijgfxy = (ix).inner(apply_gf(jy));
	return ijgfxy;
}

/// General Function to make the two electron integral <ij|g|xy>
/// For Debugging -> Expensive without intermediates
double CC_Operators::make_ijgxy(const size_t &i, const size_t &j, const real_function_3d &x, const real_function_3d &y)const{
	real_function_3d igx = (*poisson)(mo_bra_[i]*x).truncate();
	real_function_3d jy = (mo_bra_[j]*y).truncate();
	return jy.inner(igx);
}

/// General Function to make two electron integrals with pair functions (needed for energy)
double CC_Operators::make_ijgu(const size_t &i, const size_t &j, const CC_Pair &u)const{
	real_function_6d eri = TwoElectronFactory(world).dcut(parameters.lo);
	real_function_6d ij_g =
			CompositeFactory<double, 6, 3>(world).particle1(
					copy(mo_bra_[u.i])).particle2(
							copy(mo_bra_[u.j])).g12(eri);

	// compute < ij | g12 | u >
	const double ij_g_u = inner(u.function, ij_g);
	return ij_g_u;
}


/// apply the operator gf = 1/(2\gamma)*(Coulomb - 4\pi*BSH_\gamma)
/// works only if f = (1-exp(-\gamma*r12))/(2\gamma)
real_function_3d CC_Operators::apply_gf(const real_function_3d &f)const{
	double bsh_prefactor = 4.0 * constants::pi;
	double prefactor = 1.0/(2.0*corrfac.gamma());
	return prefactor*((*poisson)(f) - bsh_prefactor*(*fBSH)(f)).truncate();
}
real_function_6d CC_Operators::apply_gf(const real_function_6d &f,const size_t &particle)const{
	poisson->particle()=particle;
	fBSH->particle()=particle;
	double bsh_prefactor = 4.0 * constants::pi;
	double prefactor = 1.0/(2.0*corrfac.gamma());
	return prefactor*((*poisson)(f) - bsh_prefactor*(*fBSH)(f)).truncate();
}

}
