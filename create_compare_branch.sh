git fetch
git checkout shrestha/Add_CompilerFlag_VariableChange

mkdir temp
cp -a ./src/enable_variable_ops/* ./temp

git checkout -

cp -a temp/* ./src

git add ./src/*
git commit -am "Diff changes"

git push

