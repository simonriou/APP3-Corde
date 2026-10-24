outlets = 1;

var L = 0.65, T = 100.0, E = 3e9, r = 0.00055, rho = 0.005;
var beta = 1.4, eta = 5e-4;
var x_0 = 0.5, l_0 = 0.01, h_0 = 0.0025, x_lst = 0.41;
var N_modes = 64;

function bang() {
    var I = Math.PI * Math.pow(r, 4) / 4.0;
    
    for (var n = 1; n <= N_modes; n++) {
        var k = (n * Math.PI) / L;
        
        // Dispersion
        var term_T = (T / rho) * (k * k);
        var term_E = ((E * I) / rho) * Math.pow(k, 4);
        var omega_0 = Math.sqrt(term_T + term_E);
        
        // Damping
        var sigma = beta + 0.5 * eta * k * k;
        if (sigma >= omega_0) continue;
        
        var omega_d = Math.sqrt(omega_0 * omega_0 - sigma * sigma);
        var freq_d = omega_d / (2.0 * Math.PI);
        
        // Analytical projection A_n (Trapezoidal or Triangular pluck)
        var denom2 = L - x_0 - l_0;
        var A_n = (2.0 * h_0 / (L * k * k)) * (
            (Math.sin(k * x_0) / x_0) + 
            (Math.sin(k * (x_0 + l_0)) / denom2)
        );
        
        // Listening position weight
        var phi_lst = Math.sin(k * x_lst);
        var weight = A_n * phi_lst;
        var ratio = sigma / omega_d;

        // Output to poly~ target n, followed by the named gen~ parameters
        outlet(0, "target", n);
        outlet(0, "freq_d", freq_d);
        outlet(0, "sigma", sigma);
        outlet(0, "weight", weight);
        outlet(0, "ratio", ratio);
    }
}

// Setter functions for Max messages (e.g., "set_pluck 0.12 0.01 0.005")
function set_pluck(x0_rel, l0_val, h0_val) {
    x_0 = Math.max(0.001, x0_rel * L);
    l_0 = Math.max(0.0, l0_val);
    h_0 = h0_val;
    bang();
}