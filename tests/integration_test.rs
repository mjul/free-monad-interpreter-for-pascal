//! Integration tests for the system.
use free_monad_interpreter_for_pascal;

#[test]
fn it_can_pretty_print_all_files_in_the_pascal_directory() {
    // get the files in the pascal directory with a .pas extension
    let paths = std::fs::read_dir("pascal").unwrap();
    for path in paths {
        let path = path.unwrap().path();
        let fname = path.to_str().unwrap();

        let content = std::fs::read_to_string(fname).unwrap();

        let actual =
            free_monad_interpreter_for_pascal::pretty_print_file(&fname.to_string()).unwrap();

        // remove all whitespace from the content
        let rm_ws = |s: &String| s.chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>();
        let no_whitespace_content = rm_ws(&content);
        let no_whitespace_actual = rm_ws(&actual);

        assert_eq!(
            no_whitespace_content, no_whitespace_actual,
            "Expected files to be identical when whitespace is ignored: {}",
            fname
        );
    }
}
