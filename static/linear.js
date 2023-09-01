const ctx = document.getElementById("curve");
const ctx1 = document.getElementById("curve1");

let chart_obj = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        data: [],
        label: "Cost / Iteration Curve",
        borderColor: "#3cba9f",
        fill: false,
      },
    ],
  },
  options: {
    title: {
      display: true,
      text: "Chart JS Line Chart Example",
    },
    elements: {
      point: {
        radius: 0,
      },
    },
    responsive: true,
    maintainAspectRatio: false,
  },
});

let chart1_obj = new Chart(ctx1, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        data: [],
        label: "Y(expected) / Sample Index",
        borderColor: "#3cba9f",
        pointRadius: 2,
        fill: false,
        showLine: false,
      },
      {
        data: [],
        label: "Y(Predicted) / Sample Index",
        borderColor: "#ff9470",
        fill: false,
        pointRadius: 0,
      },
    ],
  },
  options: {
    scales: {
      y: {
          ticks: {
            // forces step size to be 50 units
            stepSize: 10,
          }
      }
    },
    responsive: true,
    maintainAspectRatio: false,
  },
});

function addData(label, newData) {
  chart_obj.data.labels.push(label);
  chart_obj.data.datasets.forEach((dataset) => {
    dataset.data.push(newData);
  });
  chart_obj.update();
}

function updateChart1(f, data)
{
  for(let i=0; i<data.length; i++)
  {
    chart1_obj.data.labels[i] = i;
    chart1_obj.data.datasets[0].data[i] = data[i][0];
    chart1_obj.data.datasets[1].data[i] = f[i];
  }
  chart1_obj.update();
}

let data = [
  [550, 2.6, 3, 20],
  [565, 3., 4, 15],
  [595, 3.6, 3, 30],
  [760, 4., 5, 8],
];
let names = ["Price", "Area", "Bedrooms", "Age"];
updateTable(data, names);
updateLearnCard();

function addGridEventListeners(td) {
  td.addEventListener("mouseover", () => {
    if (td.classList.contains("grid-clicked")) {
      return;
    }
    td.classList.add("table-active");
  });
  td.addEventListener("mouseout", () => {
    td.classList.remove("table-active");
  });
  td.addEventListener("click", () => {
    if (td.classList.contains("grid-clicked")) {
      return;
    }
    td.classList.remove("table-active");
    td.classList.add("grid-clicked");
    let input = document.createElement("input");
    input.type = "number";
    (input.pattern = "[0-9]+([.,][0-9]+)?"), (input.step = "0.1");
    input.value = td.innerHTML;
    input.addEventListener("keydown", (event) => {
      console.log(event);
      if (event.keyCode == 13 || event.keyCode == 27) {
        // enter event
        td.classList.remove("grid-clicked");
        let val = input.value;
        if (isNaN(parseFloat(val))) {
          val = "-1";
        }
        td.innerHTML = val;
        let i = td.dataset.i;
        let j = td.dataset.j;
        data[i][j] = parseFloat(val);
      }
    });
    document.addEventListener("click", (evt) => {
      let test = evt.target.closest("td");
      if (test == null || test.id != td.id) {
        td.classList.remove("grid-clicked");
        let val = input.value;
        if (isNaN(parseFloat(val))) {
          val = "-1";
        }
        td.innerHTML = val;
        let i = td.dataset.i;
        let j = td.dataset.j;
        data[i][j] = parseFloat(val);
      }
    });
    td.innerHTML = "";
    td.appendChild(input);
    input.focus();
  });
}
function addRowThEventListeners(th) {
  th.addEventListener("mouseover", () => {
    th.classList.add("bg-danger");
    th.classList.add("text-light");
  });
  th.addEventListener("mouseout", () => {
    th.classList.remove("bg-danger");
    th.classList.remove("text-light");
  });
  let original = th.innerHTML;
  let index = parseInt(original) - 1;
  th.addEventListener("click", () => {
    data.splice(index, 1);
    updateTable(data, names);
  });
}
function addNewSampleEventListeners(th) {
  th.addEventListener("mouseover", () => {
    th.classList = "text-center bg-info text-light";
  });
  th.addEventListener("mouseout", () => {
    th.classList = "text-center bg-primary text-light";
  });
  th.addEventListener("click", () => {
    data.push(new Array(names.length).fill(0));
    updateTable(data, names);
  });
}
function addEditFeatureEventListeners(th) {
  th.addEventListener("mouseover", () => {
    th.className = "bg-dark text-light";
  });
  th.addEventListener("mouseout", () => {
    th.className = "";
  });
  th.addEventListener("click", () => {
    editCard = document.getElementById("edit_feature_card");
    editCard.dataset.f_id = th.dataset.f_id;
    editCol = document.getElementById("edit_feature_col");
    editCol.classList.remove("d-none");
    newCol = document.getElementById("new_feature_col");
    newCol.classList.add("d-none");
    updateEditCard();
  });
}
function addNewFeatureEventListener(th) {
  th.addEventListener("mouseover", () => {
    th.classList.remove("bg-primary");
    th.classList.add("bg-info");
  });
  th.addEventListener("mouseout", () => {
    th.classList.remove("bg-info");
    th.classList.add("bg-primary");
  });
  th.addEventListener("click", () => {
    let col = document.getElementById("new_feature_col");
    col.classList.remove("d-none");
    editCol = document.getElementById("edit_feature_col");
    editCol.classList.add("d-none");
    updateNewCard();
  });
}
function updateEditCard() {
  let col = document.getElementById("edit_feature_col");
  let card = document.getElementById("edit_feature_card");
  let title = document.getElementById("edit_feature_title");
  let index = parseInt(card.dataset.f_id);
  title.innerHTML = names[index];
  if (index == 0) title.innerHTML += "(Y)";
  else title.innerHTML += " (X<sub>" + index + "</sub>)";
  let input = document.getElementById("edit_input");
  input.value = names[index];
  let done = document.getElementById("edit_done");
  done.replaceWith(done.cloneNode(true));
  done = document.getElementById("edit_done");
  done.addEventListener("mouseover", () => {
    done.classList.remove("bg-success");
    done.classList.add("lighter-green");
  });
  done.addEventListener("mouseout", () => {
    done.classList.remove("lighter-green");
    done.classList.add("bg-success");
  });
  done.addEventListener("click", () => {
    let st = input.value;
    if (st.length === 0) {
      st = "Invalid";
    }
    names[index] = st;
    updateTable(data, names);
    col.classList.add("d-none");
  });
  let del = document.getElementById("edit_delete");
  del.replaceWith(del.cloneNode(true));
  del = document.getElementById("edit_delete");
  del.innerHTML = "Can Not Delete (Y)";
  del.classList.remove("bg-danger");
  del.classList.remove("text-light");
  if (index != 0) {
    del.innerHTML = "Delete";
    del.classList.add("bg-danger");
    del.classList.add("text-light");
    del.addEventListener("mouseover", () => {
      del.classList.remove("bg-danger");
      del.classList.add("lighter-red");
    });
    del.addEventListener("mouseout", () => {
      del.classList.remove("lighter-red");
      del.classList.add("bg-danger");
    });
    del.addEventListener("click", () => {
      for (let i = 0; i < data.length; i++) {
        data[i].splice(index, 1);
      }
      names.splice(index, 1);
      updateTable(data, names);
      col.classList.add("d-none");
    });
  }
}
function updateNewCard() {
  let col = document.getElementById("new_feature_col");
  let create = document.getElementById("new_create");
  create.replaceWith(create.cloneNode(true));
  create = document.getElementById("new_create");
  let cancel = document.getElementById("new_cancel");
  cancel.addEventListener("mouseover", () => {
    cancel.classList.remove("bg-danger");
    cancel.classList.add("lighter-red");
  });
  cancel.addEventListener("mouseout", () => {
    cancel.classList.remove("lighter-red");
    cancel.classList.add("bg-danger");
  });
  cancel.addEventListener("click", () => {
    col.classList.add("d-none");
  });
  create.addEventListener("mouseover", () => {
    create.classList.remove("bg-success");
    create.classList.add("lighter-green");
  });
  create.addEventListener("mouseout", () => {
    create.classList.remove("lighter-green");
    create.classList.add("bg-success");
  });
  create.addEventListener("click", () => {
    let input = document.getElementById("new_name_input");
    let name = input.value;
    if (name.length === 0) name = "Invalid";
    names.push(name);
    for (let i = 0; i < data.length; i++) data[i].push(0);
    updateTable(data, names);
    col.classList.add("d-none");
  });
}
function updateLearnCard() {
  let inc_lambda = document.getElementById("inc_lambda");
  let dec_lambda = document.getElementById("dec_lambda");
  let lambda_val = document.getElementById("lambda_val");
  let inc_alpha = document.getElementById("inc_alpha");
  let dec_alpha = document.getElementById("dec_alpha");
  let alpha_val = document.getElementById("alpha_val");
  inc_alpha.addEventListener("mouseover", () => {
    inc_alpha.classList.add("bg-secondary");
    inc_alpha.classList.add("text-light");
  });
  inc_alpha.addEventListener("mouseout", () => {
    inc_alpha.classList.remove("bg-secondary");
    inc_alpha.classList.remove("text-light");
  });
  inc_alpha.addEventListener("click", () => {
    let nval = parseFloat(alpha_val.innerHTML);
    nval = (Math.round(nval * 10000) + 1) / 10000;
    alpha_val.innerHTML = nval;
  });
  dec_alpha.addEventListener("mouseover", () => {
    dec_alpha.classList.add("bg-secondary");
    dec_alpha.classList.add("text-light");
  });
  dec_alpha.addEventListener("mouseout", () => {
    dec_alpha.classList.remove("bg-secondary");
    dec_alpha.classList.remove("text-light");
  });
  dec_alpha.addEventListener("click", () => {
    let nval = parseFloat(alpha_val.innerHTML);
    nval = (Math.round(nval * 10000) - 1) / 10000;
    if (nval < 0) n_val = 0;
    alpha_val.innerHTML = nval;
  });
  inc_lambda.addEventListener("mouseover", () => {
    inc_lambda.classList.add("bg-secondary");
    inc_lambda.classList.add("text-light");
  });
  inc_lambda.addEventListener("mouseout", () => {
    inc_lambda.classList.remove("bg-secondary");
    inc_lambda.classList.remove("text-light");
  });
  inc_lambda.addEventListener("click", () => {
    let nval = parseFloat(lambda_val.innerHTML);
    nval = (Math.round(nval * 10000) + 1) / 10000;
    lambda_val.innerHTML = nval;
  });
  dec_lambda.addEventListener("mouseover", () => {
    dec_lambda.classList.add("bg-secondary");
    dec_lambda.classList.add("text-light");
  });
  dec_lambda.addEventListener("mouseout", () => {
    dec_lambda.classList.remove("bg-secondary");
    dec_lambda.classList.remove("text-light");
  });
  dec_lambda.addEventListener("click", () => {
    let nval = parseFloat(lambda_val.innerHTML);
    nval = (Math.round(nval * 10000) - 1) / 10000;
    if (nval < 0) nval = 0;
    lambda_val.innerHTML = nval;
  });
  let learn_cancel = document.getElementById("learn_cancel");
  learn_cancel.addEventListener("mouseover", () => {
    learn_cancel.classList.remove("bg-danger");
    learn_cancel.classList.add("lighter-red");
  });
  learn_cancel.addEventListener("mouseout", () => {
    learn_cancel.classList.remove("lighter-red");
    learn_cancel.classList.add("bg-danger");
  });
  learn_cancel.addEventListener("click", () => {
    let col = document.getElementById("learn_col");
    col.classList.add("d-none");
    updateTable(data, names);
  });
  let learn_start = document.getElementById("learn_start");
  learn_start.addEventListener("mouseover", () => {
    learn_start.classList.remove("bg-success");
    learn_start.classList.add("lighter-green");
  });
  learn_start.addEventListener("mouseout", () => {
    learn_start.classList.remove("lighter-green");
    learn_start.classList.add("bg-success");
  });
  learn_start.addEventListener("click", () => {
    let col = document.getElementById("learn_col");
    col.classList.add("d-none");
    let graph_col = document.getElementById("graph_col");
    graph_col.classList.remove("d-none");
    let alpha = parseFloat(alpha_val.innerHTML);
    let lambda = parseFloat(lambda_val.innerHTML);
    startLearning(alpha, lambda);
  });
}
function startLearning(alpha, lambda) {
  let graph_title = document.getElementById("graph_title");
  let stop_btn = document.getElementById("stop_btn");
  let pin_btn = document.getElementById("exit_btn");
  stop_btn.addEventListener("mouseover", () =>
  {
    stop_btn.classList.add("bg-secondary");
    stop_btn.classList.add("text-light");
  });
  stop_btn.addEventListener("mouseout", () =>
  {
    stop_btn.classList.remove("bg-secondary");
    stop_btn.classList.remove("text-light");
  });
  pin_btn.addEventListener("mouseover", () =>
  {
    pin_btn.classList.add("bg-secondary");
    pin_btn.classList.add("text-light");
  });
  pin_btn.addEventListener("mouseout", () =>
  {
    pin_btn.classList.remove("bg-secondary");
    pin_btn.classList.remove("text-light");
  });
  graph_title.innerHTML = "Learning Curve | α=" + alpha + ", λ=" + lambda;
  var socket = io();
  pin_btn.addEventListener("click", () => {
    stop_btn.innerHTML = "stop";
    socket.disconnect();
    chart_obj.data.datasets[0].data = [];
    chart_obj.data.labels = [];
    chart_obj.update();
    chart1_obj.data.datasets[0].data = [];
    chart1_obj.data.datasets[1].data = [];
    chart1_obj.data.labels = [];
    chart1_obj.update();
    let col = document.getElementById("graph_col");
    col.classList.add("d-none");
    updateTable(data,names);
  });
  socket.on("connect", function () {
    socket.emit("learn_linear", { data: data, alpha: alpha, lambda: lambda });
    let inter = setInterval(() => {
      socket.emit("get_data");
    }, 1000);
    stop_btn.addEventListener("click", () =>
    {
      if(inter == null)
      {
        inter = setInterval(() => {
          socket.emit("get_data");
        }, 1000);
        stop_btn.innerHTML = "stop";
        socket.emit("start");
        return;
      }
      clearInterval(inter);
      inter = null;
      stop_btn.innerHTML = "start";
      socket.emit("stop");
    });
  });
  socket.on("data", function (feed) {
    console.log(feed);
    addData(feed[0], feed[1]);
    updateChart1(feed[2],data);
  });
}
function updateTable(data, names) {
  let table = document.getElementById("reg-table");
  table.innerHTML = "";
  let thead = document.createElement("thead");
  table.appendChild(thead);
  let tr = document.createElement("tr");
  thead.appendChild(tr);
  let start_th = document.createElement("th");
  start_th.scope = "col";
  start_th.innerHTML = "#";
  start_th.style = "width:  5%";
  tr.appendChild(start_th);
  for (let i = 0; i < names.length; i++) {
    let th = document.createElement("th");
    th.scope = "col";
    th.innerHTML = names[i];
    if (i == 0) th.innerHTML += " (Y)";
    else th.innerHTML += " (X<sub>" + i + "</sub>)";
    th.dataset.f_id = i;
    addEditFeatureEventListeners(th);
    tr.appendChild(th);
  }
  let th = document.createElement("th");
  th.classList.add("text-center");
  th.innerHTML = "+";
  th.className = "text-center bg-primary text-light";
  th.style = "width:  5%";
  th.scope = "col";
  addNewFeatureEventListener(th);
  tr.appendChild(th);

  let tbody = document.createElement("tbody");
  table.appendChild(tbody);
  for (let i = 0; i < data.length; i++) {
    let tr = document.createElement("tr");
    tr.id = "tr_" + i;
    tbody.appendChild(tr);
    let th = document.createElement("th");
    th.scope = "row";
    th.innerHTML = "" + (i + 1);
    addRowThEventListeners(th);
    tr.appendChild(th);
    for (let j = 0; j < data[i].length + 1; j++) {
      let td = document.createElement("td");
      td.id = "td_" + i + "_" + +j;
      td.dataset.i = i;
      td.dataset.j = j;
      if (data[i].length !== j) {
        td.innerHTML = "" + data[i][j];
        addGridEventListeners(td);
      }
      tr.appendChild(td);
    }
  }
  tr = document.createElement("tr");
  tr.id = "tr_add";
  tbody.appendChild(tr);
  th = document.createElement("th");
  th.scope = "row";
  th.className = "text-center bg-primary text-light";
  th.innerHTML = "+";
  th.id = "new_sample";
  addNewSampleEventListeners(th);
  tr.appendChild(th);
  for (let j = 0; j < data[0].length; j++) {
    let td = document.createElement("td");
    tr.appendChild(td);
  }
  let done_btn = document.createElement("td");
  done_btn.classList.add("text-center");
  done_btn.className = "text-center bg-success text-light";
  done_btn.innerHTML = "Done";
  tr.appendChild(done_btn);
  done_btn.addEventListener("mouseover", () => {
    done_btn.classList.remove("bg-success");
    done_btn.classList.add("lighter-green");
  });
  done_btn.addEventListener("mouseout", () => {
    done_btn.classList.remove("lighter-green");
    done_btn.classList.add("bg-success");
  });
  done_btn.addEventListener("click", () => {
    let learn_col = document.getElementById("learn_col");
    learn_col.classList.remove("d-none");
    table.replaceWith(table.cloneNode(true));
  });
}
