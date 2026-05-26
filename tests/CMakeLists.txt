function(pde_core_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE pde_core::pde_core Catch2::Catch2WithMain)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

pde_core_add_test(test_field)
pde_core_add_test(test_mesh_block)
pde_core_add_test(test_clustering)
