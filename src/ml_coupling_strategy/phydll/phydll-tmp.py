if False:
            for i in range(num_phy_procs):
                # probably need to convert from F to C storage here?
                start = i*field_size
                end = (i+1)*field_size
                dl_input = np.array(input_field[start:end], order="F")
                dl_input = np.reshape(dl_input, input_shape)
                
                print(f"PhyDLL: dl_input for proc {i} = {dl_input}", flush=True)

                dl_output = model.predict(dl_input)
                print(f"PhyDLL: dl_output = {dl_output}")
                dl_output = dl_output[0]
                
                print(f"PhyDLL: dl_output for proc {i} = {dl_output}", flush=True)

                output_shape = np.shape(dl_output)
                output_size = np.prod(output_shape)
                dl_output = np.reshape(dl_output, (output_size))
                for k in range(output_size):
                    output_field[start + k] = dl_output[k]

            dl_fields["DL_output_field_0"] = output_field
            print(f'PhyDLL: dl_fields_0 = {dl_fields["DL_output_field_0"]}', flush=True)
            dll.send(dl_fields)

            ite += 1