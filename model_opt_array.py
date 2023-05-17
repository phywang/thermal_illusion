import mph
import numpy as np

def opt(model, matrix_mat):
    model = model.java

    model.component("comp1").geom("geom1").runPre("sel1")
    model.component("comp1").geom("geom1").feature("sel1").selection("selection").clear()

    model.component("comp1").geom("geom1").runPre("sel2")
    model.component("comp1").geom("geom1").feature("sel2").selection("selection").clear()

    for i in range(0, 40):
        for j in range(0, 40):
            if (matrix_mat[i, j].item() == 0):
                model.component("comp1").geom("geom1").feature("sel1").selection("selection").set("arr1("+str(i+1)+","+str(j+1)+")", 1)
            else:
                model.component("comp1").geom("geom1").feature("sel2").selection("selection").set("arr1("+str(i+1)+","+str(j+1)+")", 1)

    model.component("comp1").physics("ht2").feature("solid2").selection().named("geom1_sel1");


    return model


