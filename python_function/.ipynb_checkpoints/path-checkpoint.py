def to_path(task, target, model, pipeline):
    myFile = open("/code/notebook/script/task.txt", "w")
    myFile.write(task)
    myFile.close()
    myFile_1 = open("/code/notebook/script/target.txt", "w")
    myFile_1.write(target)
    myFile_1.close()
    myFile_2 = open("/code/notebook/script/model.txt", "w")
    myFile_2.write(model)
    myFile_2.close()
    myFile_3 = open("/code/notebook/script/pipeline.txt", "w")
    myFile_3.write(pipeline)
    myFile_3.close()