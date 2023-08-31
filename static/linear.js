let data = [
    [31,3213,3213,312],
    [321,343,343,3152],
    [321,343,343,3152],
  ]
  let names = ["Price", "Area", "Floors", "Stuff"]
  updateTable(data,names);

  function addGridEventListeners(td)
  {
    td.addEventListener("mouseover", () => {
      if(td.classList.contains("grid-clicked"))
      {
        return;
      }
      td.classList.add("table-active");
    });
    td.addEventListener("mouseout", () => {
      td.classList.remove("table-active");
    })
    td.addEventListener("click", () => {
      if(td.classList.contains("grid-clicked"))
      {
        return;
      }
      td.classList.remove("table-active");
      td.classList.add("grid-clicked");
      let input = document.createElement("input");
      input.type = "number";
      input.pattern = "[0-9]+([\.,][0-9]+)?",
      input.step = "0.1";
      input.value = td.innerHTML;
      input.addEventListener("keydown", (event) => {
        console.log(event);
        if(event.keyCode == 13 || event.keyCode == 27) // enter event
        {
          td.classList.remove("grid-clicked");
          let val = input.value;
          if(isNaN(parseFloat(val)))
          {
            val = "-1";
          }
          td.innerHTML = val;
          let i = td.dataset.i;
          let j = td.dataset.j;
          data[i][j] = parseFloat(val);
        }
      });
      document.addEventListener("click",(evt) => {
        let test = evt.target.closest("td");
        if(test == null || test.id != td.id)
        {
          td.classList.remove("grid-clicked");
          let val = input.value;
          if(isNaN(parseFloat(val)))
          {
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
    })
  }
  function addRowThEventListeners(th)
  {
    th.addEventListener("mouseover", () => {
      th.classList.add("bg-danger");
      th.classList.add("text-light");
    });
    th.addEventListener("mouseout", () => {
      th.classList.remove("bg-danger");
      th.classList.remove("text-light");
    });
    let original = th.innerHTML;
    let index = parseInt(original)-1;
    th.addEventListener("click", () => {
        data.splice(index,1);
        updateTable(data,names);
    });
  }
  function addNewSampleEventListeners(th)
  {
    th.addEventListener("mouseover", () => {
      th.classList = "text-center bg-info text-light";
    });
    th.addEventListener("mouseout", () => {
      th.classList = "text-center bg-primary text-light";
    });
    th.addEventListener("click", () => {
      
      data.push(new Array(names.length).fill(0));
      updateTable(data,names);
    });
  }
  function addEditFeatureEventListeners(th)
  {
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
  function addNewFeatureEventListener(th)
  {
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
    })
  }
  function updateEditCard()
  {
    let col = document.getElementById("edit_feature_col");
    let card = document.getElementById("edit_feature_card");
    let title = document.getElementById("edit_feature_title");
    let index = parseInt(card.dataset.f_id);
    title.innerHTML = names[index];
    if(index == 0) title.innerHTML += "(Y)";
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
      if(st.length === 0)
      {
        st = "Invalid";
      }
      names[index] = st;
      updateTable(data,names);
      col.classList.add("d-none");
    })
    let del = document.getElementById("edit_delete");
    del.replaceWith(del.cloneNode(true));
    del = document.getElementById("edit_delete");
    del.innerHTML = "Can Not Delete (Y)";
    del.classList.remove("bg-danger");
    del.classList.remove("text-light");
    if(index != 0)
    {
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
        for(let i=0; i<data.length; i++)
        {
          data[i].splice(index,1);
        }
        names.splice(index,1);
        updateTable(data,names);
        col.classList.add("d-none");
      });
    }
  }
  function updateNewCard()
  {
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
      if(name.length === 0) name = "Invalid";
      names.push(name);
      for(let i=0; i<data.length; i++) data[i].push(0);
      updateTable(data,names);
      col.classList.add("d-none");
    });
  }
  function updateTable(data, names)
  {
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
    for(let i=0; i<names.length; i++)
    {
      let th = document.createElement("th");
      th.scope = "col";
      th.innerHTML = names[i];
      if(i == 0) th.innerHTML += " (Y)";
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
    for(let i=0; i < data.length; i++)
    {
      let tr = document.createElement("tr");
      tr.id = "tr_" + i
      tbody.appendChild(tr);
      let th = document.createElement("th");
      th.scope="row";
      th.innerHTML=""+(i+1);
      addRowThEventListeners(th);
      tr.appendChild(th);
      for(let j=0; j < data[i].length+1; j++)
      {
        let td = document.createElement("td");
        td.id = "td_" + i + "_" + + j;
        td.dataset.i = i;
        td.dataset.j = j;
        if(data[i].length !== j)
        {
          td.innerHTML = "" + data[i][j];
          addGridEventListeners(td);
        }
        tr.appendChild(td);
      }
    }
    tr = document.createElement("tr");
    tr.id = "tr_add"
    tbody.appendChild(tr);
    th = document.createElement("th");
    th.scope="row";
    th.className = "text-center bg-primary text-light";
    th.innerHTML="+";
    th.id = "new_sample";
    addNewSampleEventListeners(th);
    tr.appendChild(th);
    for(let j=0; j < data[0].length; j++)
    {
      let td = document.createElement("td");
      tr.appendChild(td);
    }
    let done_btn = document.createElement("td");
    done_btn.classList.add("text-center");
    done_btn.className = "text-center bg-success text-light";
    done_btn.innerHTML = 'Done';
    tr.appendChild(done_btn);
  }