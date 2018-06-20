def run(host, port, config_file):
    import config
    config.load_config_file(config_file)

    global triangulation, detectionserver, fingerprint

    import localize
    import remserver
    import fingerprint

    import db
    import detectionserver
    import pageserver

    db.init()
    config.app.run(host=host, port=int(port), debug=True)
