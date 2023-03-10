const start_button = document.getElementsByClassName("start")[0];
const restart_step = document.getElementsByClassName("small-icons");
const no_epoch = document.getElementById("epoch");
const gate_select = document.getElementById("gate");
const learning_rate_select = document.getElementById("learning-rate");
const activation_select_output = document.getElementById("activation-output");
const activation_select_hidden = document.getElementById("activation-hidden");
const error_criterion_select = document.getElementById("error-criterion");
const one_zero = document.getElementById("one-zero");
const zero_zero = document.getElementById("zero-zero");
const one_one = document.getElementById("one-one");
const zero_one = document.getElementById("zero-one");
const error_section = document.getElementsByClassName("error")[0];
const answer_line = document.getElementById("answer-line");
const answer_line_x = document.getElementById("answer-line-x");
const colored_line = document.getElementById("colored-line");
const run_button = document.getElementById("run-epoch");
const select_no_epoch = document.getElementById("select-no-epoch");
const SSE_div = document.getElementById("SSE");
const MSE_div = document.getElementById("MSE");
const stop_value = document.getElementById("value-stop");
const add_stoping_value_button = document.getElementById("SSE-MSE");
const delete_stoping_value_button = document.getElementById("SSE-MSE-delete");
const is_added_SSE = document.getElementById("is-added-SSE");
const is_added_MSE = document.getElementById("is-added-MSE");
const activation_section_output = document.getElementById("activation-section-output");
const activation_section_hidden = document.getElementById("activation-section-hidden");
const input1_one = document.getElementById("input1-one");
const input1_zero = document.getElementById("input1-zero");
const input2_one = document.getElementById("input2-one");
const input2_zero = document.getElementById("input2-zero");
const input1_value = document.getElementById("input1-value");
const input2_value = document.getElementById("input2-value");
const output_value = document.getElementById("output-value");

const WHITE = "white";
const BLACK = "black";
const IS_NOT = true;
const NOT_NOT = false;
const X1 = 0;
const X2 = 1;
const Y = 2;
const INPUT1_STRING = "Input 1 : ";
const INPUT2_STRING = "Input 2 : ";
const OUTPUT_STRING = "Output  : ";

let theta = [1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2];
let sigma = [0, 0, 0];
let theta_weight = [1, -1, 1, -1, 1, -1, 1];
let delta_weights = [0, 0, 0, 0, 0, 0];
let x_theta = [0.8, -0.1, 0.3, -0.8, 0.1, -0.3];
let x_theta_weight = -1;
let learning_interval;
let error_value = [1, 1, 1, 1];
let SSE = 1;
let SSE_not = 1;
let MSE = 1;
let MSE_old = -1;
let SSE_stop = -1;
let MSE_stop = -1;
let MSE_SSE_set = false;

const GATE = {
    NOT: 0,
    AND: 1,
    NAND: 2,
    OR: 3,
    NOR: 4,
    XOR: 5,
    XNOR: 6
}

const ACTIVATION = {
    RELU: 0,
    TANH: 1,
    SIGMOID: 2,
    LINEAR: 3,
    STEP: 4
}

const ERROR_CRITERION = {
    MSE: 0,
    SSE: 1
}

let start = false;
let selected_gate = gate_select.selectedIndex;
let activation_function_output = ACTIVATION.TANH;
let activation_function_hidden = ACTIVATION.TANH;
let alpha = 0.1;
let epoch = 0;
let SSE_stop_reached = false;
let MSE_stop_reached = false;
let input1 = 0;
let input2 = 0;
let output = 0;

//neural network

const NOT_DATASET = [[0, 1], [1, 0]];
const AND_DATASET = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]];
const NAND_DATASET = [[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
const OR_DATASET = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]];
const NOR_DATASET = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]];
const XOR_DATASET = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
const XNOR_DATASET = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]];

const rand_gen = () => {
    let x = Math.floor(Math.random() * 10) / 10;
    if (Math.random() < 0.6) {
        x = -x;
    }
    return x;
}

let weights = [rand_gen(), rand_gen(), rand_gen(), rand_gen(), rand_gen(), rand_gen()];

const rand_weights = () => {
    weights = [rand_gen(), rand_gen(), rand_gen(), rand_gen(), rand_gen(), rand_gen()];
}

const linear = (value) => {
    return value;
}

const sigmoid = (value) => {
    return 1 / (1 + Math.exp(-value));
}

const tanh = (value) => {
    return 2 / (1 + Math.exp(- 2 * value)) - 1;
}

const relu = (value) => {
    if (value >= 0)
        return value;
    return 0;
}

const der_linear = (value) => {
    return 1;
}

const der_sigmoid = (value) => {
    return value * (1 - value);
}

const der_tanh = (value) => {
    return 1 - value * value;
}

const der_relu = (value) => {
    if (value >= 0)
        return 1;
    return 0;
}

const step = (value) => {
    if (value >= 0)
        return 1;
    return 0;
}

const selected_function = (value, selected, d) => {
    if (d)
        switch (selected) {
            case ACTIVATION.RELU: return der_relu(value);
            case ACTIVATION.TANH: return der_tanh(value);
            case ACTIVATION.SIGMOID: return der_sigmoid(value);
            case ACTIVATION.LINEAR: return der_linear(value);
        }
    switch (selected) {
        case ACTIVATION.RELU: return relu(value);
        case ACTIVATION.TANH: return tanh(value);
        case ACTIVATION.SIGMOID: return sigmoid(value);
        case ACTIVATION.LINEAR: return linear(value);
        case ACTIVATION.STEP: return step(value);
    }
}

const selected_dataset = (line, index) => {
    switch (selected_gate) {
        case GATE.NOT: return NOT_DATASET[line][index];
        case GATE.AND: return AND_DATASET[line][index];
        case GATE.NAND: return NAND_DATASET[line][index];
        case GATE.OR: return OR_DATASET[line][index];
        case GATE.NOR: return NOR_DATASET[line][index];
        case GATE.XOR: return XOR_DATASET[line][index];
        case GATE.XNOR: return XNOR_DATASET[line][index];
    }
}

const activation_step = (w1, w2 = 0, x1, x2 = 0, ind = -1, layer_function) => {
    if (ind == -1)
        return selected_function(x1 * w1 + x2 * w2 + theta_weight[selected_gate] * theta[selected_gate], layer_function, false);
    return selected_function(x1 * w1 + x2 * w2 + x_theta[ind] * x_theta_weight, layer_function, false);
}

const weight_traning = (x, error) => {
    return alpha * x * error;
}

const perceptron_learning = () => {
    let is_not = NOT_NOT;
    let data_size = 4
    if (selected_gate == GATE.NOT) {
        is_not = IS_NOT;
        data_size = 2;
    }

    for (let i = 0; i < data_size; i++) {
        let activation_result;
        let step_error;
        if (is_not) {
            if (SSE_not == 0) {
                clear();
                epoch--;
                return;
            }
            activation_result = activation_step(weights[0], 0, selected_dataset(i, X1), 0, -1, ACTIVATION.STEP);
            step_error = selected_dataset(i, 1) - activation_result;
        }
        else {
            activation_result = activation_step(weights[0], weights[1], selected_dataset(i, X1), selected_dataset(i, X2), -1, ACTIVATION.STEP);
            step_error = selected_dataset(i, Y) - activation_result;
        }
        // console.log("error: " + step_error + " acu : " + activation_result + " des : " + selected_dataset(i, Y) + " w1 " + weights[0] + " w2 " + weights[1]);
        weights[0] = weights[0] + weight_traning(selected_dataset(i, X1), step_error);
        if (!is_not) {
            weights[1] = weights[1] + weight_traning(selected_dataset(i, X2), step_error);
        }
        error_value[i] = step_error;
    }
    SSE = error_value[0] * error_value[0] + error_value[1] * error_value[1] + error_value[2] * error_value[2] + error_value[3] * error_value[3];
    SSE_not = error_value[0] * error_value[0] + error_value[1] * error_value[1];

}

const neural_network_learning = () => {
    let v = [0, 1, 2];
    if (selected_gate == GATE.XNOR) {
        v = [3, 4, 5];
    }
    for (let i = 0; i < 4; i++) {
        let y1 = activation_step(weights[0], weights[2], selected_dataset(i, X1), selected_dataset(i, X2), v[0], activation_function_hidden);
        let y2 = activation_step(weights[1], weights[3], selected_dataset(i, X1), selected_dataset(i, X2), v[1], activation_function_hidden);
        let y3 = activation_step(weights[4], weights[5], y1, y2, v[2], activation_function_output);
        let step_error = selected_dataset(i, Y) - y3;
        sigma[2] = selected_function(y3, activation_function_output, true) * step_error;
        sigma[1] = selected_function(y2, activation_function_hidden, true) * sigma[2] * weights[5];
        sigma[0] = selected_function(y1, activation_function_hidden, true) * sigma[2] * weights[4];
        weights[0] = weights[0] + weight_traning(selected_dataset(i, X1), sigma[0]);
        weights[1] = weights[1] + weight_traning(selected_dataset(i, X1), sigma[1]);
        weights[2] = weights[2] + weight_traning(selected_dataset(i, X2), sigma[0]);
        weights[3] = weights[3] + weight_traning(selected_dataset(i, X2), sigma[1]);
        weights[4] = weights[4] + weight_traning(y1, sigma[2]);
        weights[5] = weights[5] + weight_traning(y2, sigma[2]);
        x_theta[v[0]] = x_theta[v[0]] + weight_traning(x_theta_weight, sigma[0]);
        x_theta[v[1]] = x_theta[v[1]] + weight_traning(x_theta_weight, sigma[1]);
        x_theta[v[2]] = x_theta[v[2]] + weight_traning(x_theta_weight, sigma[2]);
        error_value[i] = step_error;
        console.log(selected_dataset(i, X1), selected_dataset(i, X2), selected_dataset(i, Y), y3, step_error);
    }
    MSE_old = MSE;
    SSE = error_value[0] * error_value[0] + error_value[1] * error_value[1] + error_value[2] * error_value[2] + error_value[3] * error_value[3];
    MSE = SSE / 4;
    SSE_div.innerHTML = "SSE = " + SSE;
    MSE_div.innerHTML = "MSE = " + MSE;

}












const edit_answer_line = () => {
    let x1, x2, y1, y2;

    if (selected_gate == GATE.XOR || selected_gate == GATE.XNOR) {
        let v = [0, 1, 2];
        if (selected_gate == GATE.XNOR) {
            v = [3, 4, 5];
        }
        x1 = -2;
        y1 = (-x_theta[v[0]] * x_theta_weight - weights[0] * x1) / weights[2];
        x2 = 2;
        y2 = (-x_theta[v[0]] * x_theta_weight - weights[0] * x2) / weights[2];

        x1 = x1 * 250 + 30;
        x2 = x2 * 250 + 30;
        y1 = 400 - 250 * y1 - 30;
        y2 = 400 - 250 * y2 - 30;

        answer_line_x.setAttribute("x1", x1);
        answer_line_x.setAttribute("y1", y1);
        answer_line_x.setAttribute("x2", x2);
        answer_line_x.setAttribute("y2", y2);

        x1 = -2;
        y1 = (-x_theta[v[1]] * x_theta_weight - weights[1] * x1) / weights[3];
        x2 = 2;
        y2 = (-x_theta[v[1]] * x_theta_weight - weights[1] * x2) / weights[3];

        x1 = x1 * 250 + 30;
        x2 = x2 * 250 + 30;
        y1 = 400 - 250 * y1 - 30;
        y2 = 400 - 250 * y2 - 30;

        answer_line.setAttribute("x1", x1);
        answer_line.setAttribute("y1", y1);
        answer_line.setAttribute("x2", x2);
        answer_line.setAttribute("y2", y2);

        return;
    }

    x1 = -2;
    y1 = ((-theta[selected_gate]) * theta_weight[selected_gate] - weights[0] * x1) / weights[1];
    x2 = 2;
    y2 = ((-theta[selected_gate]) * theta_weight[selected_gate] - weights[0] * x2) / weights[1];

    if (selected_gate == GATE.NOT) {
        y1 = -2;
        y2 = 2;
        x1 = ((-theta[selected_gate] * theta_weight[selected_gate]) / weights[0]);
        x2 = x1;
    }

    x1 = x1 * 250 + 30;
    x2 = x2 * 250 + 30;
    y1 = 400 - 250 * y1 - 30;
    y2 = 400 - 250 * y2 - 30;


    answer_line.setAttribute("x1", x1);
    answer_line.setAttribute("y1", y1);
    answer_line.setAttribute("x2", x2);
    answer_line.setAttribute("y2", y2);
    answer_line_x.setAttribute("x1", x1);
    answer_line_x.setAttribute("y1", y1);
    answer_line_x.setAttribute("x2", x2);
    answer_line_x.setAttribute("y2", y2);
}

const clear = () => {
    start = !start;
    start_button.children[0].innerHTML = '<i class="fa-sharp fa-solid fa-play fa-3x"></i>';
    start_button.children[0].classList.toggle("padding-start-icon");
    clearInterval(learning_interval);
}

const start_running = () => {
    start = !start;
    start_button.children[0].classList.toggle("padding-start-icon");
    start_button.children[0].innerHTML = "<i class='fa-solid fa-pause fa-3x'></i>";
    alpha = parseFloat(learning_rate_select.options[learning_rate_select.selectedIndex].text);
    learning_interval = setInterval(learning_loop, 10);
}

const set_points = (is_not, p0, p1, p2, p3) => {
    if (is_not) {
        one_one.classList.add("hidden");
        one_zero.classList.add("hidden");
        zero_one.style.fill = "white";
        zero_zero.style.fill = "black";
    } else {
        one_one.classList.remove("hidden");
        one_zero.classList.remove("hidden");
        one_one.style.fill = p3;
        one_zero.style.fill = p2;
        zero_one.style.fill = p1;
        zero_zero.style.fill = p0;
    }

}

const change_gate = () => {
    switch (selected_gate) {
        case GATE.NOT: set_points(IS_NOT); break;
        case GATE.AND: set_points(NOT_NOT, WHITE, WHITE, WHITE, BLACK); break;
        case GATE.NAND: set_points(NOT_NOT, BLACK, BLACK, BLACK, WHITE); break;
        case GATE.OR: set_points(NOT_NOT, WHITE, BLACK, BLACK, BLACK); break;
        case GATE.NOR: set_points(NOT_NOT, BLACK, WHITE, WHITE, WHITE); break;
        case GATE.XOR: set_points(NOT_NOT, WHITE, BLACK, BLACK, WHITE); break;
        case GATE.XNOR: set_points(NOT_NOT, BLACK, WHITE, WHITE, BLACK); break;
    }

    if (selected_gate == GATE.XNOR || selected_gate == GATE.XOR) {
        error_section.classList.remove("hidden");
    } else {
        error_section.classList.add("hidden");
    }
}

const learning_loop = () => {
    if (SSE_stop_reached || MSE_stop_reached) {
        clear();
        return;
    }
    calculate();
    // console.log(MSE_stop, MSE);
    if (selected_gate == GATE.XOR || selected_gate == GATE.XNOR) {
        activation_function_output = activation_select_output.selectedIndex;
        activation_function_hidden = activation_select_hidden.selectedIndex;
        if (MSE_stop != -1 && MSE_SSE_set) {
            if (MSE <= MSE_stop) {
                MSE_stop_reached = true;
                return;
            }
        }
        if (SSE_stop != -1 && MSE_SSE_set) {
            if (SSE <= SSE_stop) {
                SSE_stop_reached = true;
                return;
            }
        }
        neural_network_learning();
    } else {
        if (SSE == 0) {
            SSE_stop_reached = true;
            return;
        }
        perceptron_learning();
    }
    edit_answer_line();
    epoch++;
    no_epoch.innerText = epoch;
    if (Math.abs(MSE_old - MSE) < (alpha * 0.00001) && MSE > 0.1) {
        x_theta = [0.8, -0.1, 0.3, -0.8, 0.1, -0.3];
        rand_weights();
    }
}

start_button.addEventListener("click", function () {
    start = !start;
    start_button.children[0].classList.toggle("padding-start-icon");
    if (start) {
        start_button.children[0].innerHTML = "<i class='fa-solid fa-pause fa-3x'></i>";
        alpha = parseFloat(learning_rate_select.options[learning_rate_select.selectedIndex].text);
        learning_interval = setInterval(learning_loop, 10);
    } else {
        start_button.children[0].innerHTML = '<i class="fa-sharp fa-solid fa-play fa-3x"></i>';
        clearInterval(learning_interval);
    }
})

const reset = (flage = false) => {
    alpha = parseFloat(learning_rate_select.options[learning_rate_select.selectedIndex].text);
    rand_weights();
    edit_answer_line();
    epoch = 0;
    no_epoch.innerText = epoch;
    x_theta = [0.8, -0.1, 0.3, -0.8, 0.1, -0.3];
    // console.log(weights);
    SSE = 1;
    MSE = 1;
    SSE_not = 1;
    MSE_stop_reached = false;
    SSE_stop_reached = false;
    if (!flage) {
        SSE_stop = -1;
        MSE_stop = -1;
        MSE_SSE_set = false;
        is_added_MSE.innerHTML = "No stoping value added for MSE";
        is_added_SSE.innerHTML = "No stoping value added for SSE";
    }
}

restart_step[0].addEventListener("click", function () {
    //restart function
    reset(true);
})

restart_step[1].addEventListener("click", function () {
    // one step function
    alpha = parseFloat(learning_rate_select.options[learning_rate_select.selectedIndex].text);
    // console.log(alpha);
    learning_loop();
    edit_answer_line();
})

gate_select.addEventListener("change", function () {
    reset();
    selected_gate = gate_select.selectedIndex;
    epoch = 0;
    no_epoch.innerText = epoch;
    x_theta = [0.8, -0.1, 0.3, -0.8, 0.1, -0.3];
    change_gate();
    rand_weights();
    edit_answer_line();
    if (selected_gate == GATE.XOR || selected_gate == GATE.XNOR) {
        activation_section_output.classList.remove("hidden");
        activation_section_hidden.classList.remove("hidden");
    }
    else {
        activation_section_output.classList.add("hidden");
        activation_section_hidden.classList.add("hidden");
    }

})

function onlyNumberKey(evt) {
    var ASCIICode = (evt.which) ? evt.which : evt.keyCode
    if ((ASCIICode > 31 && (ASCIICode < 48 || ASCIICode > 57)) && ASCIICode != 46)
        return false;
    return true;
}

run_button.addEventListener("click", function () {
    for (let i = 0; i < select_no_epoch.value; i++) {
        learning_loop();

    }
})


add_stoping_value_button.addEventListener("click", function () {
    MSE_SSE_set = true;
    if (error_criterion_select.selectedIndex == ERROR_CRITERION.MSE) {
        MSE_stop = parseFloat(stop_value.value);
        is_added_MSE.innerHTML = "The value of MSE is : " + MSE_stop;
        return;
    }
    SSE_stop = parseFloat(stop_value.value);
    is_added_SSE.innerHTML = "The value of SSE is : " + SSE_stop;
})

delete_stoping_value_button.addEventListener("click", function () {
    if (error_criterion_select.selectedIndex == ERROR_CRITERION.MSE) {
        MSE_stop = -1;
        MSE_stop_reached = false;
        is_added_MSE.innerHTML = "No stoping value added for MSE";
    } else {
        is_added_SSE.innerHTML = "No stoping value added for SSE";
        SSE_stop_reached = false;
        SSE_stop = -1;
    }
})

learning_rate_select.addEventListener("change", function () {
    alpha = parseFloat(learning_rate_select.options[learning_rate_select.selectedIndex].text);
})

activation_select_output.addEventListener("change", function () {
    reset(true);
})

activation_select_hidden.addEventListener("change", function () {
    reset(true);
})

const calculate = () => {
    input1_value.innerHTML = INPUT1_STRING + input1;
    input2_value.innerHTML = INPUT2_STRING + input2;
    if (selected_gate == GATE.NOT) {
        output_value.innerHTML = OUTPUT_STRING + activation_step(weights[0], 0, input1, 0, -1, ACTIVATION.STEP);
    } else if (selected_gate == GATE.XOR || selected_gate == GATE.XNOR) {
        let v = [0, 1, 2];
        if (selected_gate == GATE.XNOR) {
            v = [3, 4, 5];
        }
        let y1 = activation_step(weights[0], weights[2], input1, input2, v[0], activation_function_hidden);
        let y2 = activation_step(weights[1], weights[3], input1, input2, v[1], activation_function_hidden);
        let y3 = activation_step(weights[4], weights[5], y1, y2, v[2], activation_function_output);
        y3 = (Math.round(y3 * 100) / 100).toFixed(3);
        output_value.innerHTML = OUTPUT_STRING + y3;
    } else {
        output_value.innerHTML = OUTPUT_STRING + activation_step(weights[0], weights[1], input1, input2, -1, ACTIVATION.STEP);
    }


}

input1_one.addEventListener("click", function () {
    input1 = 1;
    calculate();
})
input1_zero.addEventListener("click", function () {
    input1 = 0;
    calculate();
})
input2_one.addEventListener("click", function () {
    input2 = 1;
    calculate();
})
input2_zero.addEventListener("click", function () {
    input2 = 0;
    calculate();
})







change_gate();
